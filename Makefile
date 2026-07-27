NATIVE_DIR := native

.PHONY: all docker-build install restore clean

all: docker-build

docker-build:
	$(MAKE) -C $(NATIVE_DIR) docker-build

install: docker-build
	$(MAKE) -C $(NATIVE_DIR) install

restore:
	$(MAKE) -C $(NATIVE_DIR) restore

clean:
	$(MAKE) -C $(NATIVE_DIR) clean
