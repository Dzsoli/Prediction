from pred_3D import *
from model import *

m = nn.Conv3d(16, 33, 3, stride=1)
m = nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))
input = torch.randn(20, 16, 10, 50, 100)
output = m(input)
print(output.shape)

enc2 = Encoder_Grid2D()
print(sum(p.numel() for p in enc2.parameters() if p.requires_grad))
dec2 = Decoder_Grid2D()
print(sum(p.numel() for p in dec2.parameters() if p.requires_grad))

t = torch.randn(17,1,16,128,12)
enc = Encoder_Grid3D_3()
print(sum(p.numel() for p in enc.parameters() if p.requires_grad))
z = enc(t)
print(z.unsqueeze(4).shape)
dec = Decoder_Grid3D_3()
print(sum(p.numel() for p in dec.parameters() if p.requires_grad))
r = dec(z)
print(r.shape)

disc = Discriminator2D()

aae3d = ADVAE3D(encoder=enc, decoder=dec, discriminator=disc)
pred, z = aae3d(t)
print('aae', pred.shape, z.shape)

# tt = torch.randn(60,1,16,128)
# tt2 = tt.unsqueeze(4).reshape(5,1,16,128,12)
# print(tt[0:12])
# print(tt2[0])
# print(tt[3]-tt2[0,:,:,:,3])
#
