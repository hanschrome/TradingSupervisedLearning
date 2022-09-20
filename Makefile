help:
	@echo "build - Build the image, install requirements on image."
	@echo "run - Run the container"
build:
	@echo "Building docker image..."
	docker rm tradingSupervisedLearning || echo 'ok'
	docker build . -t trading-supervised-learning
run:
	@echo "Running..."
	docker run -it -v $$(pwd):/app trading-supervised-learning
