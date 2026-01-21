"""
Forex Advisor - Orquestrador Principal
Combina análise técnica com notícias para gerar insights contextualizados.
"""

import sys
import os
import logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import importlib.util

spec = importlib.util.spec_from_file_location("forex_scrapping", os.path.join(os.path.dirname(__file__), "src", "data", "forex-scrapping.py"))
forex_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(forex_module)
fetch_forex_data = forex_module.fetch_forex_data

from src.analysis.analysis import analyze_market

spec = importlib.util.spec_from_file_location("news_scrapping", os.path.join(os.path.dirname(__file__), "src", "news", "news-scrapping.py"))
news_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(news_module)
fetch_news_with_llm = news_module.fetch_news_with_llm

from src.agent.agent import generate_insight

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def save_analysis_to_file(analysis: dict, insight: str, output_dir: str = "outputs") -> str:
    """
    Salva o resultado da análise em um arquivo TXT.
    
    Args:
        analysis: Dicionário com resultados da análise técnica
        insight: Insight contextualizado gerado
        output_dir: Diretório onde salvar o arquivo (padrão: outputs)
    
    Returns:
        str: Caminho do arquivo gerado
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"classificacao_final_{timestamp}.txt"
    filepath = os.path.join(output_dir, filename)
    
    content_lines = []
    content_lines.append("=" * 60)
    content_lines.append("CLASSIFICAÇÃO FINAL - FOREX ADVISOR")
    content_lines.append("=" * 60)
    content_lines.append("")
    content_lines.append(f"Data/Hora da Análise: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    content_lines.append(f"Par de Moedas: BRL/USD")
    content_lines.append("")
    content_lines.append("=" * 60)
    content_lines.append("CLASSIFICAÇÃO TÉCNICA")
    content_lines.append("=" * 60)
    content_lines.append("")
    content_lines.append(f"Classificação: {analysis['classification']}")
    content_lines.append(f"Confiança: {analysis['confidence']:.1%}")
    content_lines.append(f"Explicação: {analysis['explanation']}")
    content_lines.append("")
    
    if analysis.get('feature_contributions'):
        content_lines.append("=" * 60)
        content_lines.append("FEATURES MAIS INFLUENTES")
        content_lines.append("=" * 60)
        content_lines.append("")
        top_features = sorted(
            analysis['feature_contributions'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        for i, (feature, importance) in enumerate(top_features, 1):
            content_lines.append(f"{i}. {feature}: {importance:.4f}")
        content_lines.append("")
    
    content_lines.append("=" * 60)
    content_lines.append("INSIGHT CONTEXTUALIZADO")
    content_lines.append("=" * 60)
    content_lines.append("")
    content_lines.append(insight)
    content_lines.append("")
    content_lines.append("=" * 60)
    content_lines.append("FIM DA ANÁLISE")
    content_lines.append("=" * 60)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content_lines))
        logger.info(f"Análise salva em: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Erro ao salvar arquivo: {str(e)}")
        raise


def main():
    """Função principal que orquestra todo o pipeline."""
    
    print("=" * 60)
    print("FOREX ADVISOR - Análise de BRL/USD")
    print("=" * 60)
    print()
    
    try:
        # 1. Buscar dados OHLC
        logger.info("Etapa 1/4: Buscando dados históricos de BRL/USD...")
        print("Buscando dados históricos...")
        forex_data = fetch_forex_data(years=5)
        print(f"✓ Dados coletados: {len(forex_data)} registros")
        print(f"  Período: {forex_data['Date'].min().date()} até {forex_data['Date'].max().date()}")
        print()
        
        # 2. Executar análise técnica e classificação
        logger.info("Etapa 2/4: Executando análise técnica...")
        print("Analisando indicadores técnicos...")
        analysis = analyze_market(forex_data)
        print(f"✓ Classificação: {analysis['classification']}")
        print(f"  Confiança: {analysis['confidence']:.1%}")
        print(f"  Explicação: {analysis['explanation']}")
        print()
        
        # 3. Buscar notícias recentes
        logger.info("Etapa 3/4: Buscando notícias recentes...")
        print("Buscando contexto de notícias...")
        
        # Determinar provedor de LLM
        llm_provider = os.getenv("LLM_PROVIDER", "gemini")
        if llm_provider == "gemini" and not os.getenv("GOOGLE_API_KEY"):
            logger.warning("GOOGLE_API_KEY não encontrada. Usando fallback para notícias.")
            llm_provider = "fallback"
        
        news_list = fetch_news_with_llm(
            currency_pair="BRL/USD",
            days=7,
            llm_provider=llm_provider
        )
        print(f"✓ Notícias coletadas: {len(news_list)} itens")
        for i, news in enumerate(news_list[:3], 1):
            print(f"  {i}. {news.get('title', 'Sem título')}")
        print()
        
        # 4. Gerar insights via agente LLM
        logger.info("Etapa 4/4: Gerando insights...")
        print("Gerando insight contextualizado...")
        
        # Usar mesmo provedor de LLM para insights
        insight = generate_insight(
            classification={
                'classification': analysis['classification'],
                'confidence': analysis['confidence'],
                'explanation': analysis['explanation']
            },
            indicators_summary=analysis['indicators_summary'],
            news_list=news_list,
            llm_provider=llm_provider if llm_provider != "fallback" else "fallback"
        )
        print("Insight gerado")
        print()
        
        # 5. Exibir resultado formatado
        print("=" * 60)
        print("RESULTADO FINAL")
        print("=" * 60)
        print()
        
        print("CLASSIFICAÇÃO TÉCNICA")
        print(f"   {analysis['classification']} (Confiança: {analysis['confidence']:.1%})")
        print(f"   {analysis['explanation']}")
        print()
        
        if analysis['feature_contributions']:
            print("FEATURES MAIS INFLUENTES")
            top_features = sorted(
                analysis['feature_contributions'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            for feature, importance in top_features:
                print(f"   {feature}: {importance:.4f}")
            print()
        
        print("INSIGHT CONTEXTUALIZADO")
        print(f"   {insight}")
        print()
        
        print("=" * 60)
        print(f"Análise concluída em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # 6. Salvar resultado em arquivo TXT
        logger.info("Salvando resultado em arquivo TXT...")
        filepath = None
        try:
            filepath = save_analysis_to_file(analysis, insight)
            print(f"\nResultado salvo em: {filepath}")
        except Exception as e:
            logger.error(f"Erro ao salvar arquivo: {str(e)}")
            print(f"\nAviso: Não foi possível salvar o arquivo: {str(e)}")
        
        return {
            'classification': analysis['classification'],
            'confidence': analysis['confidence'],
            'insight': insight,
            'indicators': analysis['indicators_summary'],
            'news_count': len(news_list),
            'filepath': filepath
        }
    
    except KeyboardInterrupt:
        logger.info("Interrompido pelo usuário")
        print("\n\nProcesso interrompido pelo usuário.")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Erro no pipeline principal: {str(e)}", exc_info=True)
        print(f"\n\nErro: {str(e)}")
        print("Verifique os logs para mais detalhes.")
        sys.exit(1)


if __name__ == "__main__":
    main()
