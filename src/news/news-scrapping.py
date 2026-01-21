"""
Módulo de coleta de notícias e contexto recente sobre BRL/USD.
Usa LLM para buscar e resumir notícias relevantes.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


def fetch_news_with_llm(currency_pair="BRL/USD", days=7, llm_provider="gemini"):
    """
    Busca notícias recentes usando LLM para buscar contexto.
    
    Args:
        currency_pair (str): Par de moedas (padrão: BRL/USD)
        days (int): Número de dias para buscar notícias (padrão: 7)
        llm_provider (str): Provedor de LLM
    
    Returns:
        List[Dict]: Lista de notícias com título, data e snippet
    """
    try:
        if llm_provider == "gemini":
            return _fetch_news_gemini(currency_pair, days)
        else:
            logger.warning(f"Provedor {llm_provider} não suportado. Usando fallback.")
            return _fetch_news_fallback(currency_pair, days)
    except Exception as e:
        logger.error(f"Erro ao buscar notícias com LLM: {str(e)}")
        return _fetch_news_fallback(currency_pair, days)


def _fetch_news_gemini(currency_pair, days):
    """Busca notícias usando Google Gemini via LangChain."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.prompts import ChatPromptTemplate
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("GOOGLE_API_KEY não encontrada. Usando fallback.")
            return _fetch_news_fallback(currency_pair, days)
        
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        
        prompt_text = f"""Você é um assistente especializado em análise financeira. 
Busque e resuma as principais notícias e eventos recentes (últimos {days} dias) 
que possam impactar o par de moedas {currency_pair}.

Retorne uma lista de 3-5 notícias relevantes, cada uma com:
- Título/Manchete
- Data aproximada (se disponível)
- Resumo breve (1-2 frases) explicando como pode impactar {currency_pair}

Formato de resposta (JSON):
{{
    "news": [
        {{
            "title": "Título da notícia",
            "date": "Data aproximada",
            "snippet": "Resumo do impacto"
        }}
    ]
}}

Se não encontrar notícias específicas, mencione eventos econômicos gerais relevantes 
para o par de moedas que foi informado que possam afetar a taxa de câmbio."""

        prompt_template = ChatPromptTemplate.from_messages([
            ("human", "{user_prompt}")
        ])
        
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0.5,
                max_output_tokens=800,
            )
            
            chain = prompt_template | llm
            response = chain.invoke({"user_prompt": prompt_text})
            
        except Exception as e:
            logger.warning(f"Modelo {model_name} não disponível: {str(e)}. Tentando fallback para gemini-1.5-flash.")
            try:
                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    google_api_key=api_key,
                    temperature=0.5,
                    max_output_tokens=800,
                )
                chain = prompt_template | llm
                response = chain.invoke({"user_prompt": prompt_text})
            except Exception:
                logger.warning("gemini-1.5-flash não disponível. Usando fallback.")
                return _fetch_news_fallback(currency_pair, days)
        
        content = response.content if hasattr(response, 'content') else str(response)
        
        import json
        import re
        
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                news_data = json.loads(json_match.group())
                return news_data.get("news", [])
            except json.JSONDecodeError:
                logger.warning("Erro ao parsear JSON da resposta do Gemini. Usando parser alternativo.")
                return _parse_llm_response(content, currency_pair)
        else:
            return _parse_llm_response(content, currency_pair)
    
    except ImportError:
        logger.warning("Biblioteca langchain-google-genai não instalada. Execute: pip install langchain-google-genai")
        return _fetch_news_fallback(currency_pair, days)
    except Exception as e:
        logger.error(f"Erro ao usar Google Gemini via LangChain: {str(e)}")
        return _fetch_news_fallback(currency_pair, days)


def _parse_llm_response(content: str, currency_pair: str) -> List[Dict]:
    """Parseia resposta do LLM em formato estruturado."""
    news_items = []
    
    lines = content.split('\n')
    current_item = {}
    
    for line in lines:
        line = line.strip()
        if not line:
            if current_item:
                news_items.append(current_item)
                current_item = {}
            continue
        
        if line and len(line) < 150 and not line.startswith(' '):
            if current_item:
                news_items.append(current_item)
            current_item = {'title': line, 'date': 'Recent', 'snippet': ''}
        elif current_item and 'snippet' in current_item:
            current_item['snippet'] += ' ' + line
    
    if current_item:
        news_items.append(current_item)
    
    if not news_items:
        news_items.append({
            'title': f'Contexto sobre {currency_pair}',
            'date': 'Recent',
            'snippet': content[:300]
        })
    
    return news_items[:5]

def _fetch_news_fallback(currency_pair, days):
    """
    Fallback: retorna notícias genéricas quando LLM não está disponível.
    Em produção, isso poderia fazer scraping de RSS feeds.
    """
    logger.info("Usando fallback para notícias (LLM não disponível)")
    
    return [
        {
            'title': f'Monitoramento de {currency_pair}',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'snippet': f'Análise técnica sugere monitorar indicadores econômicos do Brasil e Estados Unidos que podem impactar a taxa de câmbio {currency_pair}.'
        },
        {
            'title': 'Indicadores Econômicos',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'snippet': 'Recomenda-se acompanhar dados de inflação, taxa de juros e balança comercial de ambos os países para entender movimentos do câmbio.'
        }
    ]


def format_news_for_prompt(news_list: List[Dict]) -> str:
    """
    Formata lista de notícias para inclusão no prompt do LLM.
    
    Args:
        news_list: Lista de notícias
    
    Returns:
        str: Texto formatado
    """
    if not news_list:
        return "Nenhuma notícia recente disponível."
    
    formatted = "Notícias e contexto recente:\n"
    for i, news in enumerate(news_list, 1):
        formatted += f"{i}. {news.get('title', 'Sem título')}\n"
        formatted += f"   Data: {news.get('date', 'N/A')}\n"
        formatted += f"   {news.get('snippet', 'Sem resumo')}\n\n"
    
    return formatted


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    news = fetch_news_with_llm(currency_pair="BRL/USD", days=7)
    print("\n=== Notícias Coletadas ===")
    for item in news:
        print(f"\nTítulo: {item.get('title')}")
        print(f"Data: {item.get('date')}")
        print(f"Resumo: {item.get('snippet')}")
