from src.data_loader import load_all_data

def main():
    data = load_all_data()

    print("DATA LOADED SUCCESSFULLY\n")
    for k, v in data.items():
        if hasattr(v, "shape"):
            print(f"{k}: {v.shape}")
        else:
            print(f"{k}: loaded")

if __name__ == "__main__":
    main()
