# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: object_detection/protos/top_k_anchor_matcher.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='object_detection/protos/top_k_anchor_matcher.proto',
  package='object_detection.protos',
  serialized_pb=_b('\n2object_detection/protos/top_k_anchor_matcher.proto\x12\x17object_detection.protos\"R\n\x11TopKAnchorMatcher\x12\x1b\n\x0fnumber_of_top_k\x18\x01 \x01(\x05:\x02\x32\x30\x12 \n\x13unmatched_threshold\x18\x02 \x01(\x02:\x03\x30.3')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_TOPKANCHORMATCHER = _descriptor.Descriptor(
  name='TopKAnchorMatcher',
  full_name='object_detection.protos.TopKAnchorMatcher',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='number_of_top_k', full_name='object_detection.protos.TopKAnchorMatcher.number_of_top_k', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=20,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='unmatched_threshold', full_name='object_detection.protos.TopKAnchorMatcher.unmatched_threshold', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=0.3,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=79,
  serialized_end=161,
)

DESCRIPTOR.message_types_by_name['TopKAnchorMatcher'] = _TOPKANCHORMATCHER

TopKAnchorMatcher = _reflection.GeneratedProtocolMessageType('TopKAnchorMatcher', (_message.Message,), dict(
  DESCRIPTOR = _TOPKANCHORMATCHER,
  __module__ = 'object_detection.protos.top_k_anchor_matcher_pb2'
  # @@protoc_insertion_point(class_scope:object_detection.protos.TopKAnchorMatcher)
  ))
_sym_db.RegisterMessage(TopKAnchorMatcher)


# @@protoc_insertion_point(module_scope)