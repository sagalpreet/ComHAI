init:
	pip install -r requirements.txt --no-cache-dir

run:
	python3 experiment.py --accuracies 0.80 0.77 0.75 0.72 0.70 0.67 0.65 0.63 0.60 0.57 0.55 0.53 0.50 --model_names cnn_data resnet-110
	python3 experiment.py --accuracies 0.80 0.75 0.70 0.65 0.60 0.55 0.50 --model_names cnn_data resnet-110
	python3 experiment.py --accuracies 0.80 0.75 0.70 0.65 --model_names cnn_data resnet-110
	python3 experiment.py --accuracies 0.7 0.7 0.7 0.7 0.7 --model_names cnn_data resnet-110
	python3 experiment.py --accuracies 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 --model_names cnn_data resnet-110
	python3 experiment.py --accuracies 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 --model_names cnn_data resnet-110