import React, { useState,useRef } from 'react';
import axios from 'axios';
import Button from '@mui/material/Button';
import DeleteIcon from '@mui/icons-material/Delete';
import SendIcon from '@mui/icons-material/Send';
import Stack from '@mui/material/Stack';


const MAX_COUNT = 50

const ImageManagement = () => {
	const [msg, setMsg] = useState(null);

	const [uploadedFiles, setUploadedFiles] = useState([])
	const [fileLimit, setFileLimit] = useState(false);
	
	const uploaded = [...uploadedFiles];

	const hiddenFileInput = useRef(""); 

	const handleClick = event => {
		hiddenFileInput.current.click();   
	  };

	const handleUpload = async () => {
		uploadedFiles.forEach(async (file) => {
			const formData = new FormData();
			formData.append('file', file);
			await axios.post('/upload/', formData);
		});
		setMsg(`Number of uploaded files: ${uploadedFiles.length}`);
	};

	const handleClear = async () => {
		const response = await axios.get('/clear/');
		setMsg(response.data.message);
	};

	const handleUploadFiles = files => {
		let limitExceeded = false;
		files.some((file)=> {
			if(uploaded.findIndex((f)=> f.name === file.name) ===-1){
				uploaded.push(file);
				if (uploaded.length === MAX_COUNT){
					alert('You can only add a maximum of $(MAX_COUNT) files');
					setFileLimit(false);
					limitExceeded = true;
					return true;
				}
			}
		})
		if(!limitExceeded) setUploadedFiles(uploaded)
	}

	const handleFileEvent = (e) =>{
		const chosenFiles = Array.prototype.slice.call(e.target.files)
		handleUploadFiles(chosenFiles)
	}

	return (
		<div className='ImageManagement'>
			<Stack direction="row" spacing={2}>
				<Button className="button-upload" onClick={handleClick} variant="outlined" >
					Select Files
				</Button>
				<input id='fileUpload' type="file" multiple ref={hiddenFileInput} onChange={handleFileEvent} disabled={fileLimit} style={{display:'none'}}/>
				<Button onClick={handleUpload} variant="outlined" startIcon={<SendIcon />}>
					Upload
				</Button>
				<Button onClick={handleClear} variant="outlined" startIcon={<DeleteIcon />}>
					Delete
				</Button>
			</Stack>
			<div className='uploaded-files-list'>
				<p>Selected Files :</p>
				{uploadedFiles.map(file=>(
					<div>
						{file.name}
					</div>
				))}
				<div>
					<br></br>
					<p>{msg}</p>
				</div>
			</div>
		</div>
	);
};

export default ImageManagement;

