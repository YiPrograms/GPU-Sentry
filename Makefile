NATIVE_DIR := native

.PHONY: all docker-build install install-docker restore clean

all:
	$(MAKE) -C $(NATIVE_DIR) all

docker-build:
	$(MAKE) -C $(NATIVE_DIR) docker-build

install:
	$(MAKE) -C $(NATIVE_DIR) install

install-docker:
	$(MAKE) -C $(NATIVE_DIR) install-docker

restore:
	$(MAKE) -C $(NATIVE_DIR) restore

clean:
	$(MAKE) -C $(NATIVE_DIR) clean
