import onnx


def main():
    # Load the ONNX model file
    model = onnx.load("onnx/model.onnx")

    # Get the model's inputs and display information about each input
    print("Model Inputs:")
    for input in model.graph.input:
        print(f"Name: {input.name}, Shape: {input.type.tensor_type.shape.dim}")

    # Get the model's outputs and display information about each output
    print("Model Outputs:")
    for output in model.graph.output:
        print(f"Name: {output.name}, Shape: {output.type.tensor_type.shape.dim}")


if __name__ == "__main__":
    main()
