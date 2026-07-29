
import pandas as pd

def load_document(file_path):
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Only CSV or XLSX supported right now")

    documents = []
    for index, row in df.iterrows():
        text = ""
        for col in df.columns:
            text += f"{col}: {row[col]}\n"
        documents.append(text)

    return documents, df