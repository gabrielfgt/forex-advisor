"""
Módulo do agente LLM que gera insights contextualizados.
Combina análise técnica com notícias para criar insights informativos.
"""

import os
import re
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Palavras proibidas que indicam recomendações explícitas
FORBIDDEN_WORDS = [
    'compre agora', 'venda agora', 'compre', 'venda', 'invista', 'não invista',
    'é o momento de comprar', 'é o momento de vender', 'deve comprar', 'deve vender',
    'recomendo comprar', 'recomendo vender', 'sugiro comprar', 'sugiro vender',
    'buy now', 'sell now', 'you should buy', 'you should sell', 'invest now'
]


def generate_insight(
    classification: Dict,
    indicators_summary: Dict,
    news_list: List[Dict],
    llm_provider: str = "gemini"
) -> str:
    """
    Gera insight contextualizado combinando análise técnica e notícias.
    
    Args:
        classification: Dicionário com classificação e explicação
        indicators_summary: Dicionário com resumo dos indicadores
        news_list: Lista de notícias recentes
        llm_provider: Provedor de LLM a usar ('gemini', 'openai', 'anthropic', 'ollama')
    
    Returns:
        str: Insight
    """
    try:
        prompt = _build_prompt(classification, indicators_summary, news_list)
        
        if llm_provider == "gemini":
            try:
                insight = _generate_with_gemini(prompt)
                logger.debug(f"Insight gerado pelo Gemini: {len(insight)} caracteres")
            except Exception as e:
                logger.warning(f"Erro ao gerar insight com Gemini: {str(e)}. Usando fallback.")
                insight = _generate_fallback(classification, indicators_summary, news_list)
        else:
            logger.warning(f"Provedor {llm_provider} não suportado. Usando fallback.")
            insight = _generate_fallback(classification, indicators_summary, news_list)
        
        insight = _validate_insight(insight)
        
        if len(insight) < 50:
            logger.warning(f"Insight final muito curto ({len(insight)} caracteres). Usando fallback.")
            insight = _generate_fallback(classification, indicators_summary, news_list)
        
        logger.debug(f"Insight final validado: {len(insight)} caracteres")
        return insight
    
    except Exception as e:
        logger.error(f"Erro ao gerar insight: {str(e)}", exc_info=True)
        return _generate_fallback(classification, indicators_summary, news_list)


def _build_prompt(classification: Dict, indicators_summary: Dict, news_list: List[Dict]) -> str:
    """Constrói prompt estruturado para o LLM."""
    
    import sys
    import os
    import importlib.util
    
    news_module_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "news", "news-scrapping.py")
    spec = importlib.util.spec_from_file_location("news_scrapping", news_module_path)
    news_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(news_module)
    format_news_for_prompt = news_module.format_news_for_prompt
    
    news_text = format_news_for_prompt(news_list)
    
    prompt = f"""Você é um analista financeiro especializado em câmbio. Sua função é fornecer informações contextuais sobre o mercado de câmbio, SEM fazer recomendações de investimento.

IMPORTANTE: Você NUNCA deve fazer recomendações explícitas de compra, venda ou investimento. Apenas informe e contextualize o cenário atual.

Contexto Técnico:
- Classificação Atual: {classification.get('classification', 'N/A')}
- Confiança: {classification.get('confidence', 0):.1%}
- Explicação: {classification.get('explanation', 'N/A')}

Indicadores Principais:
- Preço Atual: {indicators_summary.get('price', 'N/A')}
- Média Móvel 20 dias: {indicators_summary.get('sma_20', 'N/A')}
- Média Móvel 50 dias: {indicators_summary.get('sma_50', 'N/A')}
- RSI: {indicators_summary.get('rsi', 'N/A')}
- Volatilidade: {indicators_summary.get('volatility', 'N/A')}%

{news_text}

Tarefa: Escreva um parágrafo de 3-4 frases que:
1. Explique o cenário atual de forma clara
2. Combine a classificação técnica com o contexto das notícias
3. Seja informativo e contextual, mas NUNCA faça recomendações de compra/venda
4. Use linguagem profissional mas acessível

Lembre-se: Você está apenas informando, não recomendando ações de investimento."""

    return prompt


def _generate_with_gemini(prompt: str) -> str:
    """
    Gera insight usando Google Gemini via LangChain.
    
    Args:
        prompt: Prompt completo para o LLM
    
    Returns:
        str: Insight gerado pelo Gemini
    """
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.prompts import ChatPromptTemplate
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY não encontrada. Configure a variável de ambiente GOOGLE_API_KEY no arquivo .env")
        
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        
        system_instruction_text = "Você é um analista financeiro que fornece informações contextuais sobre câmbio, sem fazer recomendações de investimento."
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_instruction_text),
            ("human", "{user_prompt}")
        ])
        
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0.7,
                max_output_tokens=2500,
            )
            
            chain = prompt_template | llm
            response = chain.invoke({"user_prompt": prompt})
            
        except Exception as e:
            logger.warning(f"Modelo {model_name} não disponível: {str(e)}. Tentando fallback para gemini-1.5-flash.")
            try:
                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    google_api_key=api_key,
                    temperature=0.7,
                    max_output_tokens=2500,
                )
                chain = prompt_template | llm
                response = chain.invoke({"user_prompt": prompt})
            except Exception:
                logger.warning("gemini-1.5-flash não disponível. Usando fallback com prompt concatenado.")
                full_prompt = f"{system_instruction_text}\n\n{prompt}"
                simple_prompt = ChatPromptTemplate.from_messages([
                    ("human", "{user_prompt}")
                ])
                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    google_api_key=api_key,
                    temperature=0.7,
                    max_output_tokens=2500,
                )
                chain = simple_prompt | llm
                response = chain.invoke({"user_prompt": full_prompt})
        
        content = None
        try:
            if hasattr(response, 'content'):
                content = response.content
                if isinstance(content, list) and len(content) > 0:
                    if hasattr(content[0], 'text'):
                        content = content[0].text
                    elif isinstance(content[0], str):
                        content = content[0]
                    else:
                        content = str(content[0])
                elif not isinstance(content, str):
                    content = str(content)
            elif hasattr(response, 'text'):
                content = response.text
            elif isinstance(response, str):
                content = response
            else:
                if hasattr(response, 'message') and hasattr(response.message, 'content'):
                    content = response.message.content
                elif hasattr(response, 'generations') and len(response.generations) > 0:
                    gen = response.generations[0]
                    if isinstance(gen, list) and len(gen) > 0:
                        content = gen[0].text if hasattr(gen[0], 'text') else str(gen[0])
                    else:
                        content = gen.text if hasattr(gen, 'text') else str(gen)
                elif hasattr(response, 'messages') and len(response.messages) > 0:
                    msg = response.messages[0]
                    if hasattr(msg, 'content'):
                        content = msg.content
                    else:
                        content = str(msg)
                else:
                    content = str(response)
        except Exception as e:
            logger.warning(f"Erro ao extrair conteúdo da resposta: {str(e)}. Tentando str(response).")
            content = str(response)
        
        if not isinstance(content, str):
            content = str(content)
        
        if not content or len(content.strip()) == 0:
            raise ValueError("Resposta do Gemini está vazia ou foi bloqueada")
        
        content = content.strip()
        
        if len(content) < 50:
            logger.warning(f"Resposta do Gemini parece muito curta ({len(content)} caracteres): {content}")
            logger.debug(f"Tipo da resposta original: {type(response)}, Atributos: {dir(response) if hasattr(response, '__dict__') else 'N/A'}")
        
        if content.endswith('...') or (len(content) > 0 and content[-1] not in '.!?' and len(content) < 100):
            logger.warning(f"Resposta do Gemini pode estar truncada. Tamanho: {len(content)} caracteres. Conteúdo: {content}")
        
        if len(content) < 100:
            logger.debug(f"Conteúdo completo da resposta: {repr(content)}")
        
        return content
    
    except ImportError:
        logger.error("Biblioteca langchain-google-genai não instalada. Execute: pip install langchain-google-genai")
        raise
    except ValueError as e:
        logger.error(f"Erro de configuração: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Erro ao usar Google Gemini via LangChain: {str(e)}")
        raise



def _generate_fallback(classification: Dict, indicators_summary: Dict, news_list: List[Dict]) -> str:
    """Gera insight de fallback quando LLM não está disponível."""
    classification_name = classification.get('classification', 'Neutro')
    explanation = classification.get('explanation', '')
    price = indicators_summary.get('price', 'N/A')
    
    insight_parts = []
    
    insight_parts.append(f"O mercado de BRL/USD apresenta uma classificação de '{classification_name}'.")
    
    if explanation:
        insight_parts.append(explanation)
    
    if price != 'N/A':
        insight_parts.append(f"O preço atual está em {price:.4f}.")
    
    if news_list:
        insight_parts.append("Eventos e notícias recentes podem estar influenciando a volatilidade do mercado.")
    else:
        insight_parts.append("Recomenda-se monitorar indicadores econômicos e notícias relevantes para entender melhor o contexto atual.")
    
    return ". ".join(insight_parts) + "."


def _validate_insight(insight: str) -> str:
    """
    Valida o insight para garantir que não contém recomendações explícitas.
    
    Args:
        insight: Texto do insight gerado
    
    Returns:
        str: Insight validado (pode ser modificado se necessário)
    """
    if not insight:
        logger.warning("Insight vazio recebido na validação.")
        return "O mercado apresenta condições que requerem monitoramento contínuo. Recomenda-se acompanhar indicadores técnicos e notícias relevantes para entender melhor o contexto atual."
    
    insight = insight.strip()
    insight_lower = insight.lower()
    
    if len(insight) < 50:
        logger.warning(f"Insight muito curto ({len(insight)} caracteres). Pode estar truncado: {insight[:100]}")
        return "O mercado apresenta condições que requerem monitoramento contínuo. Recomenda-se acompanhar indicadores técnicos e notícias relevantes para entender melhor o contexto atual."
    
    if len(insight) < 100 and not insight[-1] in '.!?':
        logger.warning(f"Insight pode estar truncado (termina sem pontuação): {insight[:100]}...")
        if len(insight) > 20:
            if not insight[-1] in '.!?':
                insight = insight + "."
        else:
            return "O mercado apresenta condições que requerem monitoramento contínuo. Recomenda-se acompanhar indicadores técnicos e notícias relevantes para entender melhor o contexto atual."
    
    for forbidden in FORBIDDEN_WORDS:
        if forbidden in insight_lower:
            logger.warning(f"Insight contém palavra proibida: {forbidden}. Aplicando correção.")
            pattern = re.compile(re.escape(forbidden), re.IGNORECASE)
            insight = pattern.sub("[informação removida]", insight)
    
    recommendation_patterns = [
        r'você (deve|precisa|pode) (comprar|vender|investir)',
        r'(é|seria) (recomendável|aconselhável) (comprar|vender|investir)',
        r'(sugiro|recomendo|aconselho) (que você )?(comprar|vender|investir)'
    ]
    
    for pattern in recommendation_patterns:
        if re.search(pattern, insight_lower):
            logger.warning(f"Insight contém padrão de recomendação. Aplicando correção.")
            insight = re.sub(pattern, "[informação contextual]", insight, flags=re.IGNORECASE)
    
    if not insight or len(insight.strip()) < 20:
        logger.warning("Insight muito curto após validação. Usando fallback.")
        return "O mercado apresenta condições que requerem monitoramento contínuo. Recomenda-se acompanhar indicadores técnicos e notícias relevantes para entender melhor o contexto atual."
    
    return insight.strip()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    classification = {
        'classification': 'Tendência de Alta',
        'confidence': 0.75,
        'explanation': 'Preço acima das médias móveis; RSI em zona neutra-alta'
    }
    
    indicators_summary = {
        'price': 5.1234,
        'sma_20': 5.1000,
        'sma_50': 5.0800,
        'rsi': 62.5,
        'volatility': 12.3
    }
    
    news_list = [
        {
            'title': 'Inflação no Brasil',
            'date': '2024-01-15',
            'snippet': 'Dados de inflação podem impactar a taxa de câmbio.'
        }
    ]
    
    insight = generate_insight(classification, indicators_summary, news_list, llm_provider="fallback")
    print("\n=== Insight Gerado ===")
    print(insight)
