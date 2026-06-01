#!/usr/bin/env bash
# Regenerate Python gRPC stubs from gyroid_service.proto.
# Run this once after installing requirements, and again whenever the .proto changes.
#
# Output:  gyroid_service_pb2.py
#          gyroid_service_pb2_grpc.py

set -euo pipefail
cd "$(dirname "$0")"

python -m grpc_tools.protoc \
    --proto_path=. \
    --python_out=. \
    --grpc_python_out=. \
    gyroid_service.proto

echo "Generated:"
echo "  gyroid_service_pb2.py"
echo "  gyroid_service_pb2_grpc.py"
