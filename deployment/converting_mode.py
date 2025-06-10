import tarfile
import os

def create_tar_gz(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

# Example usage
create_tar_gz("model.tar.gz", "model")