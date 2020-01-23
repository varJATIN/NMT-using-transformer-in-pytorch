from Translation import model
def infernce(inp):
  inp=model.embed_en(inp)
  inp=model.pos_encoder(inp)
  encoder_output = model.encoder(inp)
  targ=model.embed_fr(torch.tensor(2).reshape(1,1))
  while(True):
    tar=model.output_layer(model.decoder_out_layer(model.decoder(targ,encoder_output)))
    max_val,n=torch.max(tar.view(-1, vocab_size_fr),1)
    if(n[-1]==1 or len(targ)>len(inp)):
      break
    tar=model.embed_fr(n[-1).reshape(1,1,100)# 100 is word vector dimension
    targ=torch.cat((targ,tar),0)
  return n

inp=['the','most',' common', 'danger', 'is','politics']
target=['le', 'danger', 'le', 'plus', 'courant', 'est', 'la' ,'politique']
inp1=[]
for tok in inp:
  inp1.append(EN_TEXT.vocab.stoi [tok])
inp2 =torch.tensor(inp1).reshape(6,1)
pr_target=inference(inp2)
translated=" "
for tok1 in pr_target:
  translated.join(FR_TEXT.vocab.itos [tok1])
