from Translation import model
def infernce(inp):
  inp=model.embed_en(inp)
  inp=model.pos_encoder(inp)
  encoder_output = model.encoder(inp)
  targ=model.embed_fr(torch.tensor(2).reshape(1,1))
  while(round(model.output_final[-1,20,vocab_size_fr])!=1):
    tar=model.output_layer(model.decoder_out_layer(model.decoder(targ,encoder_output)))
    tar=model.embed_fr(np.argmax(tar)).reshape(1,1,256)
    targ=torch.cat((targ,tar),0)
  return pr_target

inp=['the','most',' common', 'danger', 'is','politics']
target=['le', 'danger', 'le', 'plus', 'courant', 'est', 'la' ,'politique']
for tok in inp:
  inp.append(EN_TEXT.vocab.stoi ['tok'])
inp_ =torch.tensor(inp[6:]).reshape(6,1)
pr_target=inference(inp_)

for tok in pr_target:
  print(FR_TEXT.vocab.itos ['tok'])
