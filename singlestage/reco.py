import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------------
# Define ReCo loss
# --------------------------------------------------------------------------------
def compute_reco_loss(rep, prob, pseudo_mask, valid_mask, strong_threshold=1.0, temp=0.5, num_queries=256, num_negatives=256):
  batch_size, num_feat, im_w_, im_h = rep.shape
  num_segments = pseudo_mask.shape[1]
  device = rep.device

  # compute valid binary mask for each pixel
  certain_pixel = (pseudo_mask * valid_mask).to(device)  # BCHW

  # permute representation for indexing: batch x im_h x im_w x feature_channel
  rep = rep.permute(0, 2, 3, 1)

  # compute prototype (class mean representation) for each class across all valid pixels
  all_feats = []
  hard_feats = []
  num_pixels = []
  protos = []
  for i in range(num_segments):
    certain_pixel_i = certain_pixel[:, i]  # select binary mask for i-th class
    if certain_pixel_i.sum() == 0:  # not all classes would be available in a mini-batch
      continue

    prob_ci = prob[:, i, :, :]
    rep_mask_hard = (prob_ci < strong_threshold) * certain_pixel_i.bool()  # select hard queries

    protos.append(torch.mean(rep[certain_pixel_i.bool()], dim=0, keepdim=True))
    all_feats.append(rep[certain_pixel_i.bool()])
    hard_feats.append(rep[rep_mask_hard])
    num_pixels.append(int(certain_pixel_i.sum().item()))

  # compute regional contrastive loss
  if len(num_pixels) <= 1:  # in some rare cases, a small mini-batch might only contain 1 or no semantic class
    return torch.tensor(0.0)
  else:
    reco_loss = torch.tensor(0.0)
    seg_proto = torch.cat(protos)
    valid_seg = len(num_pixels)
    seg_len = torch.arange(valid_seg)

    for i in range(valid_seg):
        # sample hard queries
        if len(hard_feats[i]) > 0:
          seg_hard_idx = torch.randint(len(hard_feats[i]), size=(num_queries,))
          anchor_feat_hard = hard_feats[i][seg_hard_idx]
          anchor_feat = anchor_feat_hard
        else:  # in some rare cases, all queries in the current query class are easy
          continue

        # apply negative key sampling (with no gradients)
        with torch.no_grad():
          # generate index mask for the current query class; e.g. [0, 1, 2] -> [1, 2, 0] -> [2, 0, 1]
          seg_mask = torch.cat(([seg_len[i:], seg_len[:i]]))

          # compute similarity for each negative segment prototype (semantic class relation graph)
          proto_sim = torch.cosine_similarity(seg_proto[seg_mask[0]].unsqueeze(0), seg_proto[seg_mask[1:]], dim=1)
          proto_prob = torch.softmax(proto_sim / temp, dim=0)

          # sampling negative keys based on the generated distribution [num_queries x num_negatives]
          negative_dist = torch.distributions.categorical.Categorical(probs=proto_prob)
          samp_class = negative_dist.sample(sample_shape=[num_queries, num_negatives])
          samp_num = torch.stack([(samp_class == c).sum(1) for c in range(len(proto_prob))], dim=1)

          # sample negative indices from each negative class
          negative_num_list = num_pixels[i+1:] + num_pixels[:i]
          negative_index = negative_index_sampler(samp_num, negative_num_list)

          # index negative keys (from other classes)
          negative_feat_all = torch.cat(all_feats[i+1:] + all_feats[:i])
          negative_feat = negative_feat_all[negative_index].reshape(num_queries, num_negatives, num_feat)

          # combine positive and negative keys: keys = [positive key | negative keys] with 1 + num_negative dim
          positive_feat = seg_proto[i].unsqueeze(0).unsqueeze(0).repeat(num_queries, 1, 1)
          all_feat = torch.cat((positive_feat, negative_feat), dim=1)

        seg_logits = torch.cosine_similarity(anchor_feat.unsqueeze(1), all_feat, dim=2)
        reco_loss = reco_loss + F.cross_entropy(seg_logits / temp, torch.zeros(num_queries).long().to(device))

    return reco_loss / valid_seg


def negative_index_sampler(samp_num, seg_num_list):
  negative_index = []
  for i in range(samp_num.shape[0]):
    for j in range(samp_num.shape[1]):
      negative_index += np.random.randint(low=sum(seg_num_list[:j]),
                                          high=sum(seg_num_list[:j+1]),
                                          size=int(samp_num[i, j])).tolist()
  return negative_index


def label_onehot(inputs, num_segments):
  batch_size, im_h, im_w = inputs.shape
  # remap invalid pixels (-1) into 0, otherwise we cannot create one-hot vector with negative labels.
  # we will still mask out those invalid values in valid mask
  # inputs = torch.relu(inputs)
  outputs = torch.zeros([batch_size, num_segments, im_h, im_w]).to(inputs.device)
  return outputs.scatter_(1, inputs.unsqueeze(1), 1.0)
