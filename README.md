# TensorFlow Models

Go to models/research/object_detection/

I found out that for almost all small objects only one anchor is assigned in Faster Rcnn and SSD.
This can be seen from files in IOU. (models/research/object_detection/IOU)

To change this I changed the strategy of assigning anchors. Originally papers used IOU > 0.7. Instead of this I used top k anchors to assign to a ground truth.

Check files (models/research/object_detection/):
protos/top_k_anchor_matcher.proto
anchor_generators/multiple_grid_anchor_generator.py
builders/anchor_generator_builder.py
builders/matcher_builder.py
core/target_assigner.py
meta_architectures/faster_rcnn_meta_arch.py
protos/anchor_generator.proto
protos/matcher.proto