# MiniEp4 - Branch divergence
## MAC0219
### Felipe Castro de Noronha - 10737032

---

Neste MiniEp tivemos como objetivo estudar o fenomeno de *branch divergence* que ocorre nas GPUs. Esse fenomeno ocorre quando temos quando temos condições (`if`, `else`, etc) dentro do kernel de uma GPU, quando isso ocorre, todas as threads de um *swarp* da GPU irão executar o kernel até o final, até que a ultima thread esteja trabalhando, descantando o trabalho inutil realizado pelas outras threds.

A maneira que encontrei para reduzir o *branch divergence* é explicada a seguir.

## Dividindo o trabalho

Nossa execução de um kernel realiza um trabalho muito laborioso sob uma array `arr[]`. Para isso, cada thread de um bloco, que possui um `idx` unico, cuida da posição `arr[idx]`. Além disso, o kernel possui dois tipos diferentes de trabalhos que pode realizar, ambos são baseados em um *loop* e a condição de escolha do trabalho depende se `arr[idx]` é `> 0.5` ou se é `<= 0.5`.

Ao executarmos os kernel sem nenhuma preocupação com o *branch divergence*, como é feito em `gpu_work_v1`, fazemos que ambos trablahos sejam realizados por todas as threads, o que desperdiça recursos da GPU.

Para contornar este problema, dividi o trabalho entre dois kernels diferentes. O kernel `gpu_work_less` cuida dos calculos realizados para a condição `arr[idx] <= 0.5` e o kernel `gpu_work_great` cuida da condição `arr[idx] > 0.5`. Além disso, cada kernel é lançado com 20 blocos e 512 threads por bloco, garantindo assim, a cobertura de toda a array `arr`.

Essa *approach* faz com que nenhum dos kernels fique ocioso, ou seja, fique fazendo trabalho que sera descartado depois, aproveitando assim, todo o potencial da GPU. Ademais, como as chamdas dividem o total de blocos e duplicam o numero de threads por bloco, asseguramos que o mesmo potencial da GPU usado na `gpu_work_v1` será utilizado, uma vez que a divisão de threads por blocos fica em cargo da GPU.

## Funciona na CPU?

Um questionamento surge quase intuitivamente, a tecnica empregada produz o mesmo resultado sendo executada em paralelo em uma CPU? O motivo para o qual aplicamos essa tecnica na GPU é a presença de *warps*, que são constituidas por 32 threads, que executam codigo em SIMT, isso é o que torna necessaria nossa otimização, para diminuir a quantdade de threads que ficam ocisosas. Como uma CPU multicore executa um codigo no regime MIMT (multiple instructions multiples threads), tal otimização não resultaria em nenhuma melhora pratica na CPU.
