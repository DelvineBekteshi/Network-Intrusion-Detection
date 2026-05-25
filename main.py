from src.preprocessing import load_and_preprocess_data

if __name__ == "__main__":
    print("Starting Data Preprocessing Pipeline")
    
    dataset_path = "archive/KDDTrain+.txt"
    
    
    X_train, X_test, y_train, y_test = load_and_preprocess_data(dataset_path)
    
   
    print("\nPreprocessing Completed Successfully!")
    print(f"X_train shape (scaled matrices): {X_train.shape}")
    print(f"X_test shape (scaled matrices): {X_test.shape}")
    print("\nTarget label distribution (Training Set):")
    print(y_train.value_counts())