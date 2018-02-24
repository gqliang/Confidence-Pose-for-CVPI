require 'paths'
paths.dofile('util.lua')
paths.dofile('img.lua')

-- Some path dierctory
Imgpath = '/home/svc2/3D_Pose/2D_ht/data/'
heatpath = '/home/svc2/3D_Pose/2D_ht/heatmap/'
-- Load pre-trained model

m = torch.load('umich-stacked-hourglass.t7')

--------------------------------------------------------------------------------
-- Main loop
--------------------------------------------------------------------------------
dataset = {'s05_seq2'}
display = false
for i, v in pairs(dataset) do       --dataset
    cam, camnum = pathdir(Imgpath .. v)
    table.sort(cam)
    for ic = 1,camnum do             --camera view
    	campath = v .. '/' .. cam[ic] .. '/'
    	os.execute("mkdir -p " .. heatpath .. campath)
        per, pernum = pathdir(Imgpath .. campath)
        table.sort(per)
        for iper = 1,pernum do        --person
                perpath = campath .. per[iper]
                frm, frmnum = pathdir(Imgpath .. perpath)
                table.sort(frm)
		-- read every frame
		preds = torch.Tensor(frmnum,16,64,64)  -- store heatmap
		print(heatpath .. perpath .. '.h5')
                for ifr = 1,frmnum    do--frame

					-- Crop the input image, which is related to the input images
					local img = image.load(Imgpath ..  perpath .. '/' .. frm[ifr])
					local newimg = image.scale(img,128,256)
					local inp = torch.zeros(img:size()[1],256,256)
					inp:sub(1,newimg:size()[1],1,256,64,191):copy(newimg:sub(1,newimg:size()[1],1,256,1,128))
					--print(Imgpath .. perpath .. '/' .. frm[ifr])
					--image.save('test.jpg',(inp))
					-- Get network output
					local out = m:forward(inp:view(1,3,256,256):cuda())
					cutorch.synchronize()
					local hm = out[#out][1]:float()
					hm[hm:lt(0)] = 0

					-- store the heatmaps
					preds[ifr]:copy(hm)
					-- Display the result
					if display then
						-- Get predictions (hm and img refer to the coordinate space)
						local preds_hm, preds_img = getPreds(hm, center, scale)
						preds_hm:mul(4) -- Change to input scale
						local dispImg = drawOutput(inp, hm, preds_hm[1])
						w = image.display{image=dispImg,win=w}
						sys.sleep(3)
					end

                end
				-- Save prediction
				local predFile = hdf5.open(heatpath .. perpath .. '.h5', 'w')
				predFile:write('preds', preds)
				predFile:close()

				collectgarbage()
            end
        end
end
if display then
	w.window:close()  -- for closing the dislay window
end
