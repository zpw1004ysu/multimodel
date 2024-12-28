# train.py

import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.optim import Adam
from config import *
from utils import set_seed, load_data, create_graph_data, evaluate_metrics, write_predictions, save_matrix
from models import FusionModel


def train_model():
    set_seed(SEED)

    # Load data
    X_train, y_train, scaler = load_data(TRAIN_PATH)
    X_test, y_test, _ = load_data(TEST_PATH, scaler)

    # Create graph data
    train_data = create_graph_data(X_train, y_train, THRESHOLD)
    test_data = create_graph_data(X_test, y_test, THRESHOLD)

    # Initialize model
    fusion_model = FusionModel(
        input_size=X_train.shape[1],
        gat_hidden_size=HIDDEN_SIZE,
        cnn_hidden_size=HIDDEN_SIZE,
        transformer_hidden_size=HIDDEN_SIZE,
        output_size=OUTPUT_SIZE,
        dropout=DROPOUT
    )

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(fusion_model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        fusion_model.train()
        optimizer.zero_grad()
        out = fusion_model(train_data, X_train)
        loss = criterion(out, train_data.y)
        loss.backward()
        optimizer.step()

        train_pred = out.argmax(dim=1)
        train_acc = accuracy_score(train_data.y.numpy(), train_pred.numpy())

        fusion_model.eval()
        with torch.no_grad():
            test_out = fusion_model(test_data, X_test)
            test_pred = test_out.argmax(dim=1)
            test_acc = accuracy_score(y_test.numpy(), test_pred.numpy())

        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

    # Final evaluation
    test_metrics = evaluate_metrics(y_test.numpy(), test_pred.numpy())
    print(f"Final Test Metrics: {test_metrics}")

    # Save outputs
    write_predictions(OUTPUT_PREDICTIONS, test_pred.numpy())
    save_matrix(OUTPUT_CONF_MATRIX, test_metrics["confusion_matrix"])
