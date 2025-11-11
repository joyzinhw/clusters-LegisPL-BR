import pandas as pd
import re
import unicodedata
from typing import List, Optional
import spacy
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
import nltk

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nlp = spacy.load('pt_core_news_lg')
    USE_LEMMATIZATION = True
except OSError:
    print("Modelo spaCy não encontrado. Usando stemmer como fallback.")
    USE_LEMMATIZATION = False
    stemmer = RSLPStemmer()

class LegislativeTextCleaner:
    """Limpeza e normalização de textos do domínio legislativo brasileiro."""
    
    def __init__(self, use_lemmatization: bool = USE_LEMMATIZATION):
        self.use_lemmatization = use_lemmatization
        
        self.stopwords = set(stopwords.words('portuguese'))
        self.legislative_stopwords = {
            'lei', 'dispõe', 'altera', 'acrescenta', 'institui', 'revoga',
            'da', 'do', 'das', 'dos', 'câmara', 'senado', 'projeto',
            'proposição', 'trâmite', 'tramitacao', 'apresentação',
            'autor', 'comissão', 'federal', 'nacional', 'brasil',
            'brasileiro', 'pública', 'público', 'estabelece', 'determina'
        }
        self.all_stopwords = self.stopwords.union(self.legislative_stopwords)
        
        self.patterns = {
            'lei_numero': re.compile(r'\b(lei|lc|pec|pl|lei\s*nº|lei\s*n\.|nº)\s*[\d\.\-\/]+', re.IGNORECASE),
            'artigo': re.compile(r'\b(art\.?|artigo|§+)\s*\d+[A-Za-z]?', re.IGNORECASE),
            'data_barra': re.compile(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b'),
            'data_iso': re.compile(r'\b\d{4}-\d{2}-\d{2}\b'),
            'url': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            'numero_generico': re.compile(r'\b\d+[º°ª]\b'),  # Remove 1º, 2ª, etc.
        }
    
    def normalize_unicode(self, text: str) -> str:
        """Normaliza unicode para NFKC."""
        return unicodedata.normalize('NFKC', text)
    
    def remove_patterns(self, text: str) -> str:
        """Remove padrões legislativos específicos."""
        for pattern in self.patterns.values():
            text = pattern.sub(' ', text)
        return text
    
    def clean_punctuation(self, text: str) -> str:
        """Remove pontuação supérflua, mantendo apenas espaços."""
        text = re.sub(r'[^\w\s-]', ' ', text)
        text = re.sub(r'\s-\s', ' ', text)
        return text
    
    def remove_extra_spaces(self, text: str) -> str:
        """Remove espaços múltiplos."""
        return re.sub(r'\s+', ' ', text).strip()
    
    def tokenize(self, text: str) -> List[str]:
        """Tokeniza o texto em palavras."""
        return text.split()
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords."""
        return [t for t in tokens if t not in self.all_stopwords and len(t) > 2]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Lematiza tokens usando spaCy."""
        doc = nlp(' '.join(tokens))
        return [token.lemma_ for token in doc if not token.is_stop]
    
    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """Aplica stemming usando RSLP."""
        return [stemmer.stem(token) for token in tokens]
    
    def clean_text(self, text: str, return_tokens: bool = False) -> str | List[str]:
        """
        Pipeline completo de limpeza.
        
        Args:
            text: Texto a ser limpo
            return_tokens: Se True, retorna lista de tokens; se False, retorna string
        
        Returns:
            Texto limpo (string ou lista de tokens)
        """
        if pd.isna(text) or not isinstance(text, str):
            return [] if return_tokens else ''
        
        text = self.normalize_unicode(text)
        

        text = text.lower()

        text = self.remove_patterns(text)
        
        text = self.clean_punctuation(text)
        
        text = self.remove_extra_spaces(text)
        
        tokens = self.tokenize(text)
        
        tokens = self.remove_stopwords(tokens)
        
        if self.use_lemmatization:
            tokens = self.lemmatize_tokens(tokens)
        else:
            tokens = self.stem_tokens(tokens)
        
        if return_tokens:
            return tokens
        else:
            return ' '.join(tokens)


dataset = pd.read_excel(
    'legis/LegisPL-BR-main/dados/projetos_PL_2022_2024_completo.xlsx',
    engine="openpyxl"
)

print(f"Total de registros originais: {len(dataset)}")

dataset = dataset.drop_duplicates(subset=['ementa'], keep='first')
print(f"Total após remoção de duplicatas: {len(dataset)}")

cleaner = LegislativeTextCleaner()

print("\nLimpando ementas...")
dataset['ementa_limpa'] = dataset['ementa'].apply(
    lambda x: cleaner.clean_text(x, return_tokens=False)
)

dataset['ementa_tokens'] = dataset['ementa'].apply(
    lambda x: cleaner.clean_text(x, return_tokens=True)
)

dataset = dataset[dataset['ementa_limpa'].str.len() > 0]
print(f"Total após remoção de ementas vazias: {len(dataset)}")

print("\n" + "="*80)
print("EXEMPLOS DE LIMPEZA")
print("="*80)
for i in range(min(3, len(dataset))):
    print(f"\nEXEMPLO {i+1}:")
    print(f"Original: {dataset.iloc[i]['ementa'][:200]}...")
    print(f"Limpo: {dataset.iloc[i]['ementa_limpa'][:200]}...")
    print(f"Tokens: {dataset.iloc[i]['ementa_tokens'][:15]}")

output_file = 'projetos_PL_2022_2024_limpo.xlsx'
dataset.to_excel(output_file, index=False, engine='openpyxl')
print(f"\nDataset limpo salvo em: {output_file}")

print("\n" + "="*80)
print("ESTATÍSTICAS")
print("="*80)
dataset['num_tokens'] = dataset['ementa_tokens'].apply(len)
print(f"Média de tokens por ementa: {dataset['num_tokens'].mean():.1f}")
print(f"Mediana de tokens: {dataset['num_tokens'].median():.1f}")
print(f"Min tokens: {dataset['num_tokens'].min()}")
print(f"Max tokens: {dataset['num_tokens'].max()}")