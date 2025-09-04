from modules.training import data_preparation, llm_trainer

if __name__ == "__main__":
    training_dataset = data_preparation.load_training_data()
    llm_trainer.train_model_cpu(training_dataset)
