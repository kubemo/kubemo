PROTO_PATH=proto
PROTO_OUT_PYTHON=kubemo/proto

.PHONY: proto-py
proto-py:
	@python -m grpc_tools.protoc \
	--proto_path $(PROTO_PATH) \
	--python_out $(PROTO_OUT_PYTHON) \
	--grpc_python_out $(PROTO_OUT_PYTHON) \
	inference.proto


.PHONY: proto-go
proto-go:
	@protoc --proto_path $(PROTO_PATH) \
	--go_out . \
	--go-grpc_out . \
	inference.proto