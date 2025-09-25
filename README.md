# 🔍 Sistema de Análise de Sentimentos

Um sistema completo de análise de sentimentos usando OpenAI GPT para processar respostas de pesquisas e gerar relatórios detalhados com insights e tendências.

## 📁 Estrutura do Projeto

```
sentiment-analysis/
├── src/                          # Código fonte principal
│   ├── __init__.py
│   ├── config_manager.py         # Gerenciamento de configuração
│   ├── excel_reader.py           # Leitor de arquivos Excel
│   ├── models.py                 # Modelos de dados
│   ├── openai_client.py          # Cliente da API OpenAI
│   ├── report_generator.py       # Gerador de relatórios
│   └── sentiment_analyzer.py     # Analisador principal
├── tests/                        # Testes unitários e de integração
│   ├── __init__.py
│   ├── test_config_manager.py
│   ├── test_excel_reader.py
│   ├── test_error_handling.py
│   ├── test_main_integration.py
│   ├── test_models.py
│   ├── test_openai_client.py
│   ├── test_performance.py
│   ├── test_report_generator.py
│   └── test_sentiment_analyzer.py
├── scripts/                      # Scripts utilitários
│   └── benchmark.py              # Benchmarks de performance
├── data/                         # Arquivos de dados
│   ├── samples/
├── output/                       # Resultados gerados
├── docs/                         # Documentação e resultados
├── .env.example                  # Exemplo de configuração
├── main.py                       # Ponto de entrada principal
├── requirements.txt              # Dependências Python
└── README.md                     # Este arquivo
```

## 🚀 Instalação e Configuração

### 1. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 2. Configurar API Key

```bash
# Copiar arquivo de exemplo
cp .env.example .env

# Editar .env e adicionar sua chave da OpenAI
OPENAI_API_KEY=sua_chave_aqui
OPENAI_MODEL=gpt-4o-mini
```

### 3. Preparar Dados

Coloque seu arquivo Excel com as respostas em `data/respostas.xlsx` ou especifique o caminho com `-f`.

## 💻 Como Usar

### Análise Básica

```bash
python main.py
```

### Análise com Arquivo Específico

```bash
python main.py -f data/respostas.xlsx
```

### Processamento Async (Mais Rápido)

```bash
python main.py --async-processing --batch-size 10
```

### Processamento Otimizado para Memória

```bash
python main.py --memory-optimized --chunk-size 100
```

### Validação Apenas

```bash
python main.py --validate-only
```

## 🔧 Opções Avançadas

```bash
# Salvar logs detalhados
python main.py --log-file logs/analysis.log -v

# Limitar número de respostas (para testes)
python main.py --max-responses 50

# Não salvar arquivos, apenas exibir no console
python main.py --no-save

# Não mostrar exemplos no console
python main.py --no-samples
```

## 📊 Benchmarks de Performance

```bash
# Executar benchmarks completos
python scripts/benchmark.py

# Executar testes de performance
python -m pytest tests/test_performance.py -v
```

## 🧪 Executar Testes

```bash
# Todos os testes
python -m pytest tests/ -v

# Testes específicos
python -m pytest tests/test_sentiment_analyzer.py -v

# Testes de performance
python -m pytest tests/test_performance.py -v
```

## 📈 Recursos

### Modos de Processamento

1. **Padrão**: Processamento síncrono confiável
2. **Async**: Processamento concorrente de alta performance
3. **Memory-Optimized**: Processamento eficiente para grandes datasets

### Otimizações

- ⚡ **Cache inteligente**: Evita análises duplicadas
- 🔄 **Retry automático**: Lida com falhas de API
- 📊 **Progress tracking**: Acompanhamento em tempo real
- 🧠 **Gestão de memória**: Otimizado para grandes volumes

### Outputs

- 📋 **Console**: Relatório em tempo real
- 📄 **JSON**: Resultados detalhados estruturados
- 📊 **TXT**: Relatório formatado para humanos
- 📈 **Estatísticas**: Performance e cache metrics

## 🎯 Exemplos de Uso

### Análise Rápida

```bash
python main.py -f data/samples/100.xlsx --async-processing
```

### Análise de Grande Volume

```bash
python main.py -f data/respostas.xlsx --memory-optimized --chunk-size 200
```

### Debug e Desenvolvimento

```bash
python main.py --validate-only --log-file debug.log -v
```

## 🔍 Troubleshooting

### Erro de API Key

```bash
# Verificar se a chave está configurada
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('OPENAI_API_KEY'))"
```

### Erro de Modelo

Certifique-se de que o modelo no `.env` está correto:
S `gpt-4o-mini` (alternativa)
- `gpt-3.5-turbo` (padrão)

### Problemas de Performance

```bash
# Executar benchmarks para diagnosticar
python scripts/benchmark.py
```

## 📝 Contribuição

1. Faça fork do projeto
2. Crie uma branch para sua feature
3. Execute os testes: `python -m pytest tests/ -v`
4. Faça commit das mudanças
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob licença MIT. Veja o arquivo LICENSE para detalhes.

## 🆘 Suporte

Para dúvidas ou problemas:

1. Verifique a documentação acima
2. Execute os testes de diagnóstico
3. Consulte os logs de erro
4. Abra uma issue no repositório

---

**Desenvolvido com ❤️ para análise eficiente de sentimentos**