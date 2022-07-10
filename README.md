### 🇺🇸  Sorry, but this README was written in Portuguese!  🇺🇸

# ![Workflow status badge](https://github.com/ewertones/corujo-assets/actions/workflows/main.yml/badge.svg) Corujo Assets -  https://corujo.com.br

## O que é isso?

https://corujo.com.br é um site de previsão de ativos da bolsa. Este repositório é responsável por coletar e atualizar os dados. Para isso, diariamente, *jobs* no Cloud Scheduler disparam uma Cloud Function no GCP (Google Cloud Platform) com os parâmetros de interesse, que puxam os dados da API da [Alphavantage](https://www.alphavantage.co/).

Após os dados serem recebidos como JSON, é feito um *parse* com o [Pandas](https://pandas.pydata.org/) para transformá-los em *Dataframes* e posteriormente enviados para o banco de dados, onde está sendo utilizado o Big Query. No momento, usuários podem visualizar os valores processados através de relatórios criados no [**DATA STUDIO**](https://datastudio.google.com/reporting/3a2ade9e-079d-443f-8302-3a76843e94ef).

O diagrama de arquitetura pode ser visualizado conforme figura abaixo:

![Diagrama de Arquitetura](https://raw.githubusercontent.com/ewertones/corujo-assets/main/docs/Architecture%20Diagram.svg)

Caso seja de interesse, você pode também visualizar o [BACK-END](https://github.com/ewertones/corujo-backend) e o [FRONT-END](https://github.com/ewertones/corujo-frontend) da aplicação feito com FastAPI + React.

## Suporte

Caso precise de alguma autorização, ou deseja algum tipo de suporte para acessar/clonar/testar o projeto, mande um e-mail para admin@corujo.com.br relatando o problema.