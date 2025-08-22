# TCC_IA_Training
Primeiros Passos:
- Instalar docker e docker compose
- Se for linux, rodar no bash: sudo usermod -aG docker $USER
Após isso, ir no diretório raiz do projeto (TCC_IA_TRAINING) e rodar o comando:
docker-compose -f devconteiner/docker-compose.yml up --build (Para subir o conteiner com a imagem atualizada)
E docker-compose -f devconteiner/docker-compose.yml exec python-dev bash para conseguir acessar dentro do container.
