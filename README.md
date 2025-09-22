GPT2 Chinese 甄嬛传&庆余年
==

1.Description:
---
从头训练一个82M的中文GPT2模型，使用BERT的Tokenizer.中文语料采用后宫甄嬛传以及庆余年小说，大小约14M。训练30个周期，batchsize=8。最终可以续写10句以上的连续文字。

2.Start:
----
(1)***environment***

首先，我们下载依赖。
```bash
pip install -r requirements.txt
```

(2)***dataset***

准备中文语料，放置在./data/文件夹下，将语料中的.txt文件使用删除换行工具转换为.json文件的格式。将语料分别存入input.json和train.json文件中。

删除换行工具参考：https://uutool.cn/nl-trim-all/

(3)***Model***

在model_config 定义初始GPT-2模型的超参数配置，
- "initializer_range": 0.02 ： 定义了模型参数（如权重矩阵）在初始化时的标准差，权重会在均值为0，标准差为0.02的正态分布中进行随机初始化。
- "layer_norm_epsilon": 1e-05 ： 用于层归一化的常数，用于避免在归一化过程中出现除以零的情况。设置值为1e-05，用于稳定训练。
- "n_ctx": 1024 ： 表示模型上下文窗口的大小，GPT-2 在生成文本时会考虑的最大序列长度。最大长度设为1024，即模型一次最多能处理1024个 token。
- "n_embd": 768 ： 表示每个token的嵌入维度大小，即模型中词向量的维度。设置为768，即每个词汇的表示向量是768维的。
- "n_head": 12 ： 表示自注意力机制中的注意力头的数量。设置为12，即模型的多头注意力机制中有12个独立的头。
- "n_layer": 10 ： 表示 Transformer 编码器中的层数。在这里，设置为 12，即模型有 12 层堆叠的 Transformer 块。
- "n_positions": 1024 ： 表示模型可以处理的最大位置索引，即序列中的最大位置数。最大位置数为 1024，和 n_ctx一致，表示模型最多能处理1024个位置的token。
- "vocab_size": 13317 ： 表示词汇表的大小，即模型可以识别和生成的词汇数量。在这里，词汇表大小为 21128，表示该模型可以处理的词汇量为21128个不同的 token。


(4)***Training***

现在，我们可以使用我们处理好的数据集来训练我们的初始gpt2模型，使用如下命令：
```bash
python train.py   --model_config config/model_config_small.json   --tokenized_data_path data/tokenized/   --tokenizer_path cache/vocab_small.txt   --raw_data_path data/train.json   --epochs 15   --log_step 200   --stride 512   --output_dir model/   --device 0,1   --num_pieces 100   --raw
```

在这个过程中，我们可以看到命令窗口打印出模型的config文件，定义了模型的结构；同时也打印出了模型的参数量，为81894144，约82M

Print Model config
config:
{
  "attn_pdrop": 0.1,
  "embd_pdrop": 0.1,
  "finetuning_task": null,
  "initializer_range": 0.02,
  "layer_norm_epsilon": 1e-05,
  "n_ctx": 1024,
  "n_embd": 768,
  "n_head": 12,
  "n_layer": 10,
  "n_positions": 1024,
  "num_labels": 1,
  "output_attentions": false,
  "output_hidden_states": false,
  "output_past": true,
  "pruned_heads": {},
  "resid_pdrop": 0.1,
  "summary_activation": null,
  "summary_first_dropout": 0.1,
  "summary_proj_to_labels": true,
  "summary_type": "cls_index",
  "summary_use_proj": true,
  "torchscript": false,
  "use_bfloat16": false,
  "vocab_size": 13317
}
number of parameters: 81894144

训练过程中，每十次epoch对应的模型都将存储在./model/目录下，最终训练好的模型将存储在./model/final_model/路径中。

(5)***Generate***

现在，我们可以使用我们用目标语料训练生成的模型来进行文字生成，使用如下命令：
```bash
python generate.py   --device 1   --length 1000   --tokenizer_path cache/vocab_small.txt   --model_path model/final_model   --prefix "[CLS]范闲十分惊恐地看着皇帝。"   --topp 1   --temperature 1.0 --save_samples --save_samples_path ./mnt/
```

3.Result
--
最终会生成10个文字样本，存储在./mnt/目录下，其中之一如下：

======================================== SAMPLE 1 ========================================

范闲十分惊恐地看着皇帝。这些天里，范闲与所有的人，都不可能猜到。因为皇帝一直没有出手。可是如同一个猜想的那样，范闲在那天下人的心中，依然有些震惊。为什么他会隐藏着一个心思？为什么这些人知道神庙的秘密，会让他一直藏在这片大海里。“苦荷是位大宗师的关门弟子。”苦荷转过身来，看着皇帝说道：“神庙是世间最顶尖刻的那个人，就是最无比的武道境界高大独，如果自己不是神庙的人，那么这个方法很虚无飘渺，那么藏在这里，自己就应该如何控制那次。而最关键的是，最后的合击力。”这些思维影子与叶流云的分析，并不见得能够帮助范闲，可是他没有想到，居然会对这件事情的结论如此清楚，和神庙有关，仅凭四顾剑曾经尝试过，可是力量中对神庙之行承认。在那日紧张的时刻，范闲终于明白了面前的所作所为，陛下这一生，只要这一次点，陛下自然会想清楚地可以就此扶持。而如今的天下，从来没有出现过。范闲不会轻视这种变化，他可以尝试着进行最直接的思议，他甚至勇敢地站直了出去，在皇帝面前，毫无疑问是断然不能够将范闲与四顾剑谈判的细节，直接把这个惊天的秘密传递给了范闲。不论苦荷是对四顾剑，还是皇帝，在动用自己手中的筹码，以四顾剑的境界，如果对方认输，范闲自然可以很遗憾的地步，可是他依然在意，因为他不知道皇帝的底牌，只要四顾剑选择了大宗师的后事，他就一定要进入东夷城的血脉，将叶流云的存在，这只手归于己的头上，再多一件更大的麻烦。“你最宠爱的四顾剑。”皇帝的唇角微翘笑了起来，笑容里有些凝重，“朕也有资格让你知道，这最麻烦的事情，朕不想改变你的局，哪怕如今的庆国只是个庸而已。”是的，当范闲强行压抑下真心与恐惧的四顾剑，陈萍萍曾经是一个站在自己这个年轻人的立场上考虑的问题，如果范闲不会出手，只怕四顾剑想用最简单的手段，替庆国陷入危险之中，庆帝也有不愿意看到这一幕。然而庆帝这位大宗师终究是把自己看成了庆国的大家，如果要当下半个骄傲，自己要登基，指望整个庆国度吗？“朕这一生，最后靠这一生，逼了这么多年，也让朕难堪设想得轻松些。”皇帝忽然冷漠开口说道：“当年你也曾经写过一句话，然而刚生在这个行为手中，逼着朕发出霸道功诀的一些气息，逼着朕，逼至今日朕的这天下，不是什么好事。”这位大宗师依旧沉默很久之后，他缓缓抬起头来，双眼里似乎浮现出无数的情绪。这位大宗师也不一样，他静静地看着他，等着他向着龙袍发起最后的那方向，轻声说道：“然而我相信，这种事情，最终还是命中对我有极大的

==========================================================================================
