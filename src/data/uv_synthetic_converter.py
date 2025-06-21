# https://www.kaggle.com/code/tomooinubushi/convert-stanford-ribonanza-rna-folding-dataset
# https://medium.com/@jgbrasier/working-with-pdb-files-in-python-7b538ee1b5e4
import os
from collections.abc import Iterable
from glob import glob
import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
from threading import Thread, Lock


class UVParser:

    def __init__(
            self,
            root_path: str,
            new_root_path: str,
            index_filepath: str,
    ):

        self.root_path: str = root_path
        self.new_root_path: str = new_root_path
        self.index_filepath: str = index_filepath
        self.index_df: pd.DataFrame = pd.DataFrame(columns=['target_id', 'seq_len', 'has_msa'])
        self.lock: Lock = Lock()

    @classmethod
    def read_pdb_to_dataframe(cls, pdb_path: str, model_index: int = 1) -> pd.DataFrame:
        """
        Read a PDB file, and return a Pandas DataFrame containing the atomic coordinates and metadata.

        Args:
            pdb_path (str, optional): Path to a local PDB file to read. Defaults to None.
            model_index (int, optional): Index of the model to extract from the PDB file, in case
                it contains multiple models. Defaults to 1.

        Returns:
            pd.DataFrame: A DataFrame containing the atomic coordinates and metadata, with one row
                per atom
        """
        atomic_df = PandasPdb().read_pdb(pdb_path)
        atomic_df = atomic_df.get_model(model_index)
        if len(atomic_df.df["ATOM"]) == 0:
            raise ValueError(f"No model found for index: {model_index}")

        return pd.concat([atomic_df.df["ATOM"], atomic_df.df["HETATM"]])

    @classmethod
    def preprocess_pdb(cls, pdb_path: str):
        df = cls.read_pdb_to_dataframe(pdb_path)
        df = df[df.atom_name == "C1'"].sort_values('residue_number').reset_index(drop=True)
        df = df[['residue_name', 'residue_number', 'x_coord', 'y_coord', 'z_coord']].copy()
        df.columns = ['resname', 'resid', 'x_1', 'y_1', 'z_1']
        df[['x_1', 'y_1', 'z_1']] = df[['x_1', 'y_1', 'z_1']].astype(np.float32)
        df['resid'] = df['resid'].astype(np.int32)
        assert len(df['resid'].unique()) == (df['resid'].max() - df['resid'].min()) + 1
        return df

    def convert_pdb(self, pdb_paths: Iterable[str], thread_id: int) -> None:
        index_df: pd.DataFrame = pd.DataFrame(columns=['target_id', 'seq_len', 'has_msa'])
        for ind, pdb_path in enumerate(pdb_paths):
            try:
                pdb_id = pdb_path.replace("\\", "/").split('/')
                pdb_id = pdb_id[-2] + '_' + pdb_id[-1].removesuffix(".pdb")
                if (ind + 1) % 10_000 == 0:
                    print(f"Thr {thread_id}: Processing {ind + 1}")
                ext_label_df = self.preprocess_pdb(pdb_path)
                ext_label_df['resid'] += 1  # Adjust residue numbering to start from 1
                ext_label_df['ID'] = [f'{pdb_id}_{resid}' for resid in ext_label_df['resid']]
                ext_label_df['target_id'] = pdb_id
                ext_label_df = ext_label_df[['ID', 'resname', 'resid', 'x_1', 'y_1', 'z_1', 'target_id']]
                ext_sequence_df = pd.DataFrame({"target_id": [pdb_id], "sequence" : [''.join(ext_label_df['resname'])]})
                ext_sequence_df.to_csv(
                    os.path.join(self.new_root_path, f"{pdb_id}.in"),
                    index=False,
                    sep=",",
                )
                ext_label_df.to_csv(
                    os.path.join(self.new_root_path, f"{pdb_id}.gt"),
                    index=False,
                    sep=",",
                )

                index_df = pd.concat(
                    [self.index_df, pd.DataFrame({
                        'target_id': [pdb_id],
                        'seq_len': [len(ext_sequence_df['sequence'].iloc[0])],
                        'has_msa': [False]
                    })],
                    ignore_index=True,
                )
            except Exception as e:
                print(f'Thr {thread_id}: Failed to read {pdb_path}. Error: {str(e)}')

        with self.lock:
            self.index_df = pd.concat([self.index_df, index_df], ignore_index=True)

    def convert(self) -> None:
        if not os.path.exists(self.new_root_path):
            os.makedirs(new_root_path)

        if os.path.exists(self.index_filepath):
            print(f"Index file already exists at {self.index_filepath}. Skipping conversion.")
            return

        pdb_paths: list[str] = glob(f"{self.root_path}/*/*.pdb")

        print(f"Converting {len(pdb_paths)} PDB files in {self.root_path}...")

        threads = []
        thread_count: int = 12
        batch_size = (len(pdb_paths) + thread_count - 1) // thread_count
        for i in range(thread_count):
            t = Thread(target=lambda: self.convert_pdb(pdb_paths[i * batch_size:(i + 1) * batch_size], i + 1))
            threads.append(t)
            t.start()

        for t in threads:  # Wait for all threads to complete
            t.join()

        self.index_df.to_csv(index_filepath, index=False, sep=",")

        print(f"Saved reformatted data")

        return


def fix_index(root_path: str, index_filepath: str):
    in_paths: list[str] = glob(os.path.join(root_path, "*.in"))
    print("Found", len(in_paths), "input files.")

    target_ids: list[str] = []
    seq_lens: list[int] = []
    for i, in_path in enumerate(in_paths):
        if (i + 1) % 10_000 == 0:
            print(f"Processing {i + 1} of {len(in_paths)}")
        row = pd.read_csv(in_path, sep=",").iloc[0]

        target_id = row['target_id']
        sequence = row['sequence']

        assert os.path.exists(os.path.join(root_path, f"{target_id}.gt")), \
            f"Ground truth file for {target_id} not found."

        target_ids.append(target_id)
        seq_lens.append(len(sequence))

    index_df = pd.DataFrame({
        'target_id': target_ids,
        'seq_len': seq_lens,
    })

    index_df.to_csv(index_filepath, index=False, sep=",")


if __name__ == "__main__":
    new_root_path = r"E:\Raw Datasets\UW Synthetic RNA structures\converted"
    index_filepath = os.path.join(new_root_path, "index.csv")

    UVParser(
        root_path=r"E:\Raw Datasets\UW Synthetic RNA structures",
        new_root_path=new_root_path,
        index_filepath=index_filepath
    ).convert()

    fix_index(new_root_path, index_filepath)
