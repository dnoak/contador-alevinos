# Contador de larvas de peixes

Software de detecção e contagem automática de larvas peixes usando machine learning.

### API:

##### Execução da API
1. Em primeiro lugar, é necessário colocar as redes treinadas nos caminhos ```data/models/<nome_da_rede>/<arquivo_da_rede>```, seguindo os caminhos previstos na classe ```ModelsPaths``` do arquivo ```src/predictor/counter.py```, sendo que as redes de mesma arquitetura (variando apenas o tamanho dela) são colocadas na mesma pasta. Obs: as redes YOLO e RT-DETR são apenas arquivos ```.pt```, enquanto a DETR e DEFORMABLE-DETR são pastas.
2. Após isso, basta apenas executar o arquivo ```src\api\app.py``` que a API estará pronta para o uso.

##### Uso da API:
1. Rotas da API:
    * ```/```: Verifica se a API está online
    * ```/set-params```: Modifica os parâmetros da classe do contador de larvas ```CounterModel```, por enquanto o único parâmetro modificável é o nome da rede. Ex:
        * POST &rarr; 
        ```json
        {"model_name": "rtdetr-x"}
        ``` 
        ```json
        {"model_name": "yolov8n"}
        ```
    * ```/get-params```: Retorna os parâmetros da classe ```CounterModel```.
    *  ```/contador-alevinos```: Rota para a contagem das larvas de peixe, recebe os seguintes argumentos em uma lista de json:
    * POST &rarr; ```/contador-alevinos```; 
    ```json
        [
            {
                "_id": "id_da_imagem",
                "image": "ZGphc2lp...",
                "grid_scale": 0.3,
                "confiance": 0.5,  
                "return_image": true,
            },
            {"args_2": "..."},
            {"args_3": "..."}, 
            {"args_n": "..."} 
        ]
    ```
    * Retorno da rota ```/contador-alevinos```:
    ```json
    {
        "results": [
            {
                "_id": "id_da_imagem",
                "grid_scale": 0.3,
                "confiance": 0.5,           
                "total_count": 104,          
                "grid_results": [              
                    {"grid_xyxy": [1,2,3,4], "grid_index": [1,2]}, {"...": "..."}
                ],
                "annotated_image": "XGphckG..."
            },
            {"res_2": "..."}, 
            {"res_3": "..."}, 
            {"res_n": "..."} 
        ]
    }
    ```


2. O arquivo ```src\api\test_api.py``` faz alguns testes automáticos na API, tanto na rota ```/contador-alevinos``` quanto nas rotas ```/set-params``` e ```/get-params```. Tem-se a opção de chamar os métodos individuais de teste, ou chamar o método ```.test_all()```, que fará um teste generalizado automático. Para fazer testes visuais e ver as anotações das imagens, use os métodos ```.test_visual_one_image()``` ou ```.test_visual_many_images()```



### Instrução de como o contador de larvas foi feito:
1. Treino das redes:
- As redes foram treinadas com cortes de 640x640 pixels de imagens de larvas de peixes. Os arquivos para o treinos são individuais para cada rede e estão em ```src/models/<nome_da_rede>/train_<nome_da_rede>```
2. Ajuste dos parâmetros de inferência das redes:
- Depois de treinar as redes, cada modelo teve os seus 3 parâmetros de inferência ```(resize_scale, grid_scale, confiance)``` permutados dentro de uma faixa para obter a melhor combinação desses 3 parâmetros com base nas métricas MAE, MAPE e RMSE. Esse passo é feito no arquivo ```utils/metrics/metrics_permutation.py```. Nele tem a classe ```ArgsPermutator```, na qual o método ```.add()``` adiciona uma nova rede com uma faixa de parâmetros para ser permutados e testados um a um dentro do dataset escolhido. No fim é gerado um arquivo com os melhores parâmetros na pasta ```resuls/params_comparison/```
