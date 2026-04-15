import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import torch
from torch_geometric.data import Data

def generate_integrated_cdrs(filepath="data/mock_police_cdrs_integrated.csv"):
    NUM_USERS = 1000
    ANOMALOUS_NODES = list(range(1, 21))
    NORMAL_NODES = list(range(21, 1000))
    HIGH_FREQUENCY_NODE = 999

    calls_data = []
    start_date = datetime.strptime('2026-01-01', '%Y-%m-%d')
    end_date = datetime.strptime('2026-01-31', '%Y-%m-%d')
    TOWER_IDS = [f"TWR_{str(i).zfill(3)}" for i in range(1, 51)]
    high_risk_towers_ids = ['TWR_001', 'TWR_002', 'TWR_003']

    def random_date(start, end, night_only=False):
        delta = end - start
        random_second = random.randrange((delta.days * 24 * 60 * 60) + delta.seconds)
        dt = start + timedelta(seconds=random_second)
        if night_only: 
            dt = dt.replace(hour=random.randint(1, 3))
        return dt

    for node in ANOMALOUS_NODES:
        for _ in range(25):
            receiver = random.choice([n for n in ANOMALOUS_NODES if n != node])
            calls_data.append([node, receiver, random_date(start_date, end_date, night_only=True), random.randint(10, 45), random.choice(high_risk_towers_ids), 1])
        for _ in range(5):
            receiver = random.choice(NORMAL_NODES)
            calls_data.append([node, receiver, random_date(start_date, end_date), random.randint(30, 600), random.choice(TOWER_IDS), 1])

    for _ in range(7000):
        caller, receiver = random.sample(NORMAL_NODES, 2)
        calls_data.append([caller, receiver, random_date(start_date, end_date), random.randint(30, 1800), random.choice(TOWER_IDS), 0])

    for _ in range(300):
        receiver = random.choice(NORMAL_NODES)
        calls_data.append([HIGH_FREQUENCY_NODE, receiver, random_date(start_date, end_date), random.randint(30, 60), random.choice(TOWER_IDS), 0])

    df = pd.DataFrame(calls_data, columns=['Caller_ID', 'Receiver_ID', 'Timestamp', 'Duration_Seconds', 'Tower_ID', 'Label'])
    df = df.sort_values(by='Timestamp').reset_index(drop=True)
    df.to_csv(filepath, index=False)
    return df

def preprocess_for_gat(df):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['is_night'] = df['Timestamp'].dt.hour.apply(lambda x: 1 if 1 <= x <= 4 else 0)
    df['is_high_risk_tower'] = df['Tower_ID'].apply(lambda x: 1 if x in ['TWR_001', 'TWR_002', 'TWR_003'] else 0)

    unique_nodes = pd.concat([df['Caller_ID'], df['Receiver_ID']]).unique()
    node_mapping = {node: i for i, node in enumerate(unique_nodes)}
    num_nodes = len(unique_nodes)

    source_nodes = df['Caller_ID'].map(node_mapping).values
    target_nodes = df['Receiver_ID'].map(node_mapping).values
    edge_index = torch.tensor(np.array([source_nodes, target_nodes]), dtype=torch.long)

    feature_list = []
    inverse_mapping = {v: k for k, v in node_mapping.items()}
    for i in range(num_nodes):
        user_calls = df[(df['Caller_ID'] == inverse_mapping[i]) | (df['Receiver_ID'] == inverse_mapping[i])]
        if len(user_calls) == 0:
            feature_list.append([0.0, 0.0, 0.0])
        else:
            norm_duration = user_calls['Duration_Seconds'].mean() / 3600.0
            night_ratio = user_calls['is_night'].mean()
            tower_ratio = user_calls['is_high_risk_tower'].mean()
            feature_list.append([norm_duration, night_ratio, tower_ratio])

    x = torch.tensor(feature_list, dtype=torch.float)

    labels = np.zeros(num_nodes)
    for orig_id, new_id in node_mapping.items():
        if orig_id in range(1, 21): labels[new_id] = 1
    y = torch.tensor(labels, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)

    indices = torch.randperm(num_nodes)
    train_size = int(0.8 * num_nodes)
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.train_mask[indices[:train_size]] = True
    data.test_mask[indices[train_size:]] = True

    return data, node_mapping, inverse_mapping
