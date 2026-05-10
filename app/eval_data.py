# eval_data.py

eval_dataset = [
    {
        "inputs": {"query": "Qual o valor da fatura do Angelo e quando vence?"},
        "outputs": "O valor é de R$ 350,00 com vencimento em 05/05/2026.",
        "expectations": {
            "expected_facts": [
                "Valor de R$ 350,00",
                "Vencimento original em 05/05/2026",
                "Nome do cliente: Angelo"
            ]
        }
    },
    {
        "inputs": {"query": "Quais as penalidades por atraso no pagamento?"},
        "outputs": "Multa de 2% e juros de 1% ao mês.",
        "expectations": {
            "expected_facts": [
                "Multa fixa de 2% sobre o montante",
                "Juros de mora de 1% ao mês",
                "Cálculo pro rata die"
            ]
        }
    },
    {
        "inputs": {"query": "Quanto custa o plano Ultra Fiber e quais as vantagens?"},
        "outputs": "Custa R$ 199,00, tem 600 Mega e instalação prioritária.",
        "expectations": {
            "expected_facts": [
                "Velocidade de 600 Mega",
                "Custo mensal de R$ 199,00",
                "Instalação prioritária",
                "Classificação como Cliente Premium"
            ]
        }
    },
    {
        "inputs": {"query": "Esqueci a senha do meu roteador, qual a padrão?"},
        "outputs": "A senha padrão é Connect@2026.",
        "expectations": {
            "expected_facts": [
                "Senha padrão: Connect@2026",
                "Deve ser alterada obrigatoriamente no primeiro acesso"
            ]
        }
    },
    {
        "inputs": {"query": "O que acontece se eu atrasar a conta por mais de 30 dias?"},
        "outputs": "O sinal será interrompido totalmente.",
        "expectations": {
            "expected_facts": [
                "Suspensão total do sinal após 30 dias",
                "Suspensão parcial ocorre após 15 dias (1Mbps)"
            ]
        }
    }
]