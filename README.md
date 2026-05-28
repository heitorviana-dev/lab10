# Laboratório 10 — O Pipeline Definitivo
## RAG, QLoRA e Otimização de Inferência na GPU

> **Partes deste laboratório foram geradas/complementadas com IA, revisadas e validadas por Heitor Viana**

---

## Estrutura do Repositório

```
lab10/
├── lab10_pipeline.ipynb   # Notebook principal com todos os passos
└── README.md              # Este arquivo: análise arquitetural + métricas
```

---

## Como Executar

1. Abra o notebook no **Google Colab** (`Ambiente de execução → Alterar tipo → T4 GPU`)
2. Execute as células na ordem (Células 1 → 8)
3. O notebook detecta automaticamente a GPU e seleciona a melhor implementação de atenção disponível

### ⚠️ Nota sobre FlashAttention-2

FlashAttention-2 exige arquitetura **Ampere ou superior** (`sm_80+`, ex: A100, RTX 3090).  
A **Tesla T4** do Colab gratuito é arquitetura **Turing** (`sm_75`) e **não suporta FA2**.

O notebook implementa um **fallback automático** seguindo esta hierarquia:

```
GPU sm_80+ com flash-attn instalado  →  flash_attention_2
GPU qualquer + PyTorch ≥ 2.0         →  sdpa (Scaled Dot-Product Attention)  ← T4 usa este
CPU                                  →  eager
```

O `sdpa` usa `torch.nn.functional.scaled_dot_product_attention`, que no backend seleciona kernels otimizados (incluindo `mem_efficient_attention` da xFormers quando disponível), produzindo **resultados equivalentes** ao FA2 em termos de correção numérica e significativa economia de memória.

---

## Métricas de Benchmark

> Valores obtidos em execução no Google Colab T4 (15 GB VRAM).

| Métrica | Sem KV Cache | Com KV Cache + SDPA | Δ |
|---|---|---|---|
| Tempo de geração (100 tokens) | 375,33 s | 8,73 s | 43,0× mais rápido |
| Throughput (tokens/s) | 0,27 | 11,46 | +11,2 tok/s |
| Pico de VRAM (MB) | 5.865,3 MB | 5.715,5 MB | −2,6% |
| VRAM modelo carregado (4-bit) | 758,3 MB | 758,3 MB | Pico inicial: 814,3 MB |
| Tokens no contexto (após truncamento) | 4.096 | 4.096 | — |
| Implementação de atenção | sdpa | sdpa | — |

---

## Análise Arquitetural (Passo 5)

### Parte A — Como QLoRA + KV Cache + FlashAttention "salvaram" o Transformer

O colapso de memória observado no ambiente de produção da HealthTech tinha três causas sobrepostas, cada uma endereçada por uma técnica distinta.

O primeiro problema era o **footprint estático do modelo**: carregar o Llama-3 em Float16 consumiria entre 13 e 16 GB de VRAM apenas para os pesos, antes de processar um único token. A **quantização QLoRA em 4-bit (NF4)** resolve isso diretamente — ela representa cada peso com apenas 4 bits usando o formato NormalFloat4, que preserva a distribuição estatística dos pesos de LLMs melhor que inteiros simples, e mantém os cálculos em Float16 para precisão. O resultado é uma redução de ~75% no tamanho dos pesos, permitindo que o modelo caiba confortavelmente na VRAM disponível.

O segundo problema era o **recálculo redundante durante a decodificação**. Em um Transformer auto-regressivo, cada novo token gerado precisa de atenção sobre todos os tokens anteriores. Sem cache, isso significa recalcular os vetores K e V para *todos* os n tokens do histórico a cada passo — complexidade O(n²) no tempo. O **KV Cache** elimina esse desperdício armazenando os tensores K e V já computados na SRAM da GPU; nos passos seguintes, apenas o novo token é projetado e concatenado ao cache existente, reduzindo a fase de decodificação para complexidade O(1) por token.

O terceiro problema era a **materialização quadrática da matriz de atenção**. O cálculo padrão de `softmax(QK^T / √d) · V` cria explicitamente uma matriz n×n na HBM (memória principal da GPU), que para n=4096 em 16 camadas ocupa vários GB. O **FlashAttention-2** (ou seu equivalente SDPA no PyTorch) evita essa materialização computando a atenção em blocos que cabem na SRAM (cache L2 da GPU), que é muito mais rápida e menor. O resultado é uma dramática redução no pico de VRAM durante a fase de *prefilling* do prompt.

A combinação das três técnicas é sinérgica: QLoRA libera memória estática, SDPA/FA2 corta o pico de memória dinâmica no prefilling, e o KV Cache elimina o gargalo temporal e de memória na decodificação — juntos, tornam viável um pipeline que individualmente cada técnica não conseguiria salvar.

---

### Parte B — Por que o FlashAttention falharia com 2 milhões de tokens e por que a indústria precisa de State Space Models

Mesmo com FlashAttention-2, existe um limite fundamental que nenhuma otimização de atenção consegue superar: o **KV Cache cresce linearmente com o comprimento da sequência**. Para cada camada de atenção, o cache armazena dois tensores de tamanho `[n_tokens, n_heads, d_head]` em Float16. Com um modelo de 70B parâmetros típico (80 camadas, 64 cabeças de 128 dimensões), o KV Cache para 2 milhões de tokens requer aproximadamente:

```
2 tensores × 80 camadas × 2 milhões tokens × 64 cabeças × 128 dims × 2 bytes
≈ 320 GB apenas de cache
```

Isso excede em muito a capacidade de qualquer GPU atual (máximo ~80 GB numa H100) — e nem mesmo a comunicação entre múltiplas GPUs resolveria o problema de latência, pois cada token gerado precisaria ler todo esse cache.

O problema mais profundo é que o Transformer possui **complexidade de memória O(n)** no KV Cache e complexidade computacional **O(n²)** na atenção (que FA2 reduz em constante, não em ordem de complexidade). Para contextos ultra-longos, nenhuma implementação de atenção quadrática é viável.

É por isso que a indústria começou a migrar para **State Space Models (SSMs)** como a arquitetura **Mamba** (Gu & Dao, 2023). Mamba representa o estado do contexto como uma **matriz de estado fixo** de dimensão `d_state` (tipicamente 16 ou 64), independentemente do comprimento da sequência. Matematicamente, o SSM computa:

```
h_t = A · h_{t-1} + B · x_t   (atualização de estado: O(d_state²))
y_t = C · h_t                  (projeção de saída)
```

onde `h_t` é o vetor de estado oculto de tamanho constante. Isso significa que o **estado a ser mantido durante a geração é O(1)** em relação ao comprimento da sequência — independentemente de processar 15.000 ou 2 milhões de tokens, o modelo mantém exatamente o mesmo tamanho de estado. O custo computacional de cada novo token é constante, e não existe nada análogo ao KV Cache que cresça com n.

A limitação do Mamba é que o estado de dimensão fixa age como uma "memória seletiva com esquecimento" — ele não pode, por design, recuperar verbatim um token específico de muito tempo atrás com a exatidão que a atenção permite. Para casos de uso que exigem raciocínio preciso sobre contextos muito longos (como RAG com 30.000+ tokens e citação de trechos específicos), arquiteturas híbridas **Mamba + Atenção Esparsa** (como o Jamba da AI21 Labs) oferecem o melhor dos dois mundos: SSMs para contexto ilimitado e atenção local para precisão semântica de curto alcance. Esta é a direção que a indústria está tomando para sistemas de produção com contextos na casa dos milhões de tokens.

---

## Versionamento

Este repositório deve conter a tag `v1.0` na versão final entregue:

```bash
git tag -a v1.0 -m "Laboratório 10 — versão final"
git push origin v1.0
```

### Log de Commits

| Commit | Descrição |
|---|---|
| **Commit 1** | Setup do ambiente, carregamento do modelo em 4-bit NF4 (QLoRA), detecção de GPU com fallback SDPA, simulação do RAG massivo com ~12.000 tokens |
| **Commit 2** | Benchmark baseline: geração de 100 tokens sem KV Cache, registro de tempo e pico de VRAM |
| **Commit 3** | Otimização com KV Cache + SDPA/FlashAttention-2, tabela comparativa de métricas, README com análise arquitetural completa |

---

## Referências

- Dettmers et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. NeurIPS 2023.
- Dao et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. NeurIPS 2022.
- Dao (2023). *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*. ICLR 2024.
- Gu & Dao (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. arXiv:2312.00752.
- PyTorch (2023). *torch.nn.functional.scaled_dot_product_attention*. PyTorch 2.0 Release Notes.
