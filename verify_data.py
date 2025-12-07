from src.config import TRAIN_DATA_PATH
from src.data.loader import load_meddialog_data
from src.data.preprocessor import preprocess_dataframe
from src.data.eda import EDA
from src.data.labeler import HeuristicLabeler

def verify_data_prep():
    print("Loading data...")
    df = load_meddialog_data(TRAIN_DATA_PATH)
    df = preprocess_dataframe(df)
    
    print("Running EDA...")
    eda = EDA(df)
    eda.run()
    
    print("Running Labeler...")
    labeler = HeuristicLabeler()
    df = labeler.label_dataframe(df)
    
    print("Verification complete.")

if __name__ == "__main__":
    verify_data_prep()
