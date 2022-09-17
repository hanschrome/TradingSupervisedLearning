help:
	@echo "build - Build the image, install requirements on image."
	@echo "run - Run the container"
build:
	@echo "Building docker image..."
	docker build . -t python3-buster
run:
	@echo "Running..."
	docker run -it -v $$(pwd):/app python3-buster
