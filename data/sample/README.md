# data/sample/

Reservado para uma **amostra pontual** extraída manualmente do ZIP oficial
(`data/raw/urbanflood24_api.zip`), caso o avaliador ou o autor decida inspecionar dados reais
do dataset além da inspeção somente-leitura feita por `src/inspect_zip.py`.

Vazio por padrão nesta entrega. O script `src/inspect_zip.py` lista o conteúdo do ZIP e, se
identificar arquivos pequenos e úteis (ex.: `.csv`, `.json`, `.txt`, ou um `.npy`/`.tif` isolado
de poucos MB), **sugere** o comando de extração pontual — mas nunca extrai automaticamente.

A solução funcional desta entrega **não depende** de nenhum arquivo aqui: o pipeline principal
(`src/train_demo.py`) usa dados sintéticos gerados por `src/synthetic_data.py`, conforme
justificado no README principal do projeto.
