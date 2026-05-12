import modal

VOLUME_NAME = "debias-vlm-persistent-storage"
volume = modal.Volume.from_name(VOLUME_NAME)
stub = modal.App()

@stub.function(volumes={"/mnt/data": volume})
def list_data():
    import os
    import subprocess
    print("Listing /mnt/data:")
    subprocess.run(["ls", "-R", "/mnt/data"])

if __name__ == "__main__":
    with stub.run():
        list_data.remote()
