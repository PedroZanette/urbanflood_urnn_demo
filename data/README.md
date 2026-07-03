# data/

Esta pasta organiza os dados usados nesta entrega.

## Estrutura

- `raw/` — dataset oficial do U-RNN, baixado do Figshare, mantido **apenas como referência**.
  O arquivo `urbanflood24_api.zip` (~18 GB compactado, ~116 GB extraído) **não foi extraído
  integralmente** por limitação de armazenamento em disco local. Ver justificativa completa no
  [README principal](../README.md#justificativa-do-uso-de-dados-sintéticosreduzidos).
- `sample/` — espaço reservado para uma eventual amostra pontual extraída do ZIP oficial
  (ver `src/inspect_zip.py`). Vazio por padrão nesta entrega.
- `*.pdf`, `*.docx` — enunciado oficial da atividade (Americas TechGuard, Período 7), mantido
  localmente como referência acadêmica. Excluído do versionamento Git (`.gitignore`) por ser
  material institucional, não parte da solução técnica.

## Por que o ZIP não foi extraído

O dataset oficial do artigo U-RNN, disponibilizado no Figshare, ocupa ~18 GB compactado e
expande para ~116 GB. O ambiente de desenvolvimento usado nesta atividade não possui espaço
livre suficiente para essa extração completa. O enunciado oficial da atividade permite
explicitamente o uso de "tutoriais, quickstart, subconjuntos, exemplos reduzidos, dados
sintéticos justificados ou pesos pré-treinados" quando o dataset completo for inviável — ver
`src/inspect_zip.py` para o script que inspeciona o conteúdo do ZIP sem extraí-lo, e
`src/synthetic_data.py` para a geração dos dados sintéticos usados no pipeline funcional.
