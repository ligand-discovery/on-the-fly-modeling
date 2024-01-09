FROM python:3.10.7-bullseye

WORKDIR .

COPY . .

RUN python -m pip install --upgrade pip
RUN python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN git clone https://github.com/DhanshreeA/TabPFN.git
RUN python -m pip install -e TabPFN/.
RUN python -m pip install lolP==0.0.4
RUN python -m pip install streamlit
RUN python -m pip install networkx
RUN python -m pip install python-louvain
RUN echo "Components installation done"
RUN git clone https://github.com/ligand-discovery/fragment-embedding.git
RUN python -m pip install -e fragment-embedding/.
RUN echo "Fragment embedding installed"
RUN python -m pip install CombineMols

EXPOSE 8501
CMD ["streamlit", "run", "app/app.py"]