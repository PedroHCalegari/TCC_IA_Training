# TCC_IA_Training
Desenvolvimento:

    Para o desenvolvimento do código da IA, não será necessário utilizar o docker compose por enquanto.

    Antes de começar a mexer no código após clonar/ realizar um pull do diretório, siga esses passos:

    No terminal do diretório, criar um ambiente virtual .venv com o seguinte comando:
    python3 -m venv .venv

    Em seguida, ativar o ambiente virtual com:
    source .venv/bin/activate

    Assim, instalar as dependências já salvas em requirements.txt:
    pip install -r requirements.txt

    Quando for fazer um push no repositório, sempre lembre de acessar o ambiente virtual e atualizar as dependências salvas no requirements.txt com:
    pip freeze > requirements.txt

    Para fechar o ambiente virtual, rode:
    deactivate
