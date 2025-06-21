import torch



def main():
    if torch.cuda.is_available():
        print("CUDA is available. GPU detected:", torch.cuda.get_device_name(0))
    else:
        print("CUDA is not available. Running on CPU.")


if __name__ == "__main__":
    main()
