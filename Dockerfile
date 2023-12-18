FROM python:3.10.7-bullseye

WORKDIR .

COPY . .

RUN python -m pip install --upgrade pip
RUN python -m pip install rdkit
RUN python -m pip install streamlit
RUN python -m pip install networkx
RUN python -m pip install tabpfn
RUN python -m pip install lolP
RUN echo "Components installation done"
RUN git clone https://github.com/ligand-discovery/fragment-embedding.git
RUN cd fragment-embedding
RUN python -m pip install -e .
RUN cd ..
RUN echo "Fragment embedding installed"

EXPOSE 8501
CMD ["streamlit", "run", "app/app.py"]