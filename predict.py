from pre_processes import retrieve_train, retrieve_test, prob_to_rles, mask_to_rle, resize, np
from UNET import unet_architecture
import pandas as pd

epochs = 50

train_img, train_mask = retrieve_train()

test_img, test_img_sizes = retrieve_test()

u_net = unet_architecture()

print("\nTraining")
u_net.fit(train_img, train_mask, batch_size=16, epochs=epochs)

print("Predicting")
test_mask = u_net.predict(test_img, verbose=1)

test_mask_upsampled = []
for i in range(len(test_mask)):
    test_mask_upsampled.append(resize(np.squeeze(test_mask[i]),
                                      (test_img_sizes[i][0], test_img_sizes[i][1]),
                                      mode='constant', preserve_range=True))

test_ids, rles = mask_to_rle(test_mask_upsampled)

sub = pd.DataFrame()
sub['ImageId'] = test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('predicted_rle.csv', index=False)

print("CSV formed")


