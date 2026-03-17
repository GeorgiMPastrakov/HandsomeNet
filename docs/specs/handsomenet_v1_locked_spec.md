# HandsomeNet v1 Locked Spec

## Final lock

- Input resolution: `224x224`
- Task: single-image 2D hand pose estimation
- Dataset: FreiHAND
- Raw supervision: FreiHAND training RGB images, `training_xyz.json`, `training_K.json`
- Authoritative target derivation:
  1. Project 3D joints into image space using `K`
  2. Apply the exact same geometric preprocessing as the image
  3. Express joints in final `224x224` image coordinates
  4. Normalize to `[0, 1]`
- Final target shape: `(21, 2)`

## Joint order

```text
0  wrist
1  thumb_cmc
2  thumb_mcp
3  thumb_ip
4  thumb_tip
5  index_mcp
6  index_pip
7  index_dip
8  index_tip
9  middle_mcp
10 middle_pip
11 middle_dip
12 middle_tip
13 ring_mcp
14 ring_pip
15 ring_dip
16 ring_tip
17 pinky_mcp
18 pinky_pip
19 pinky_dip
20 pinky_tip
```

## Skeleton graph

```text
(0,1), (1,2), (2,3), (3,4),
(0,5), (5,6), (6,7), (7,8),
(0,9), (9,10), (10,11), (11,12),
(0,13), (13,14), (14,15), (15,16),
(0,17), (17,18), (18,19), (19,20)
```

## Architecture

`224x224 RGB -> mobile-style CNN backbone -> (B, 192, 14, 14) -> joint token extraction -> (B, 21, 128) -> 2 graph-attention layers -> shared 128 -> 64 -> 2 coordinate head -> bounded normalized (B, 21, 2)`

## First implementation checkpoint

Verify the full FreiHAND dataset integration contract before serious model training:

- annotation load path
- 3D-to-2D projection
- geometry-preserving preprocessing
- normalized target generation
- visual overlays
- inverse mapping correctness

