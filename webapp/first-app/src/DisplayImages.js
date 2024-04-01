import React, { useEffect, useState } from 'react';
import axios from 'axios';
import nullUrl from "./null.webp" 
import Button from '@mui/material/Button';
import Stack from '@mui/material/Stack';

const DisplayImages = () => {
    const [images, setImages] = useState([]);
    const [dir, setDir] = useState('./uploads');

    useEffect(() => {
        
    });

    const handleGenerate = async () => {
        console.log({dir})
        const response = await axios.post('/generate/',dir);
        if (Array.isArray(response.data)) {
			console.log(Array.isArray(response.data))
            setImages(response.data);
        } else {
			console.log(Array.isArray(response.data))
			setImages(nullUrl);

        }
    };

	const changeTargetDir = async () => {
        if (dir == './uploads'){
            setDir('./strip_reader');
        }
		else{
            setDir('./uploads');
        }
	};

    return (
        <div>
            <Stack direction="row" spacing={2}>
                <Button onClick={handleGenerate} variant="outlined">
                    Generate Results
                </Button>
                <Button onClick={changeTargetDir} variant="outlined">
                    Change Target Directory
                </Button>
                <p>Target Directory : {dir}</p>
            </Stack>
            <div style={{ display: 'flex', flexWrap: 'wrap' }}>
                {images !=nullUrl ? (
                    images.map((image, index) => (
                        <div key={index} style={{ width: '50%', boxSizing: 'border-box', padding: '5px' }}>
                            <img src={`data:image/jpeg;base64,${image}`} alt={`Result ${index + 1}`} style={{ width: '100%', height: 'auto' }} />
                        </div>
                    ))
                ) : (
                    <div style={{ width: '50%', boxSizing: 'border-box', padding: '5px' }}>
                        <img src={images} style={{ width: '100%', height: 'auto' }} />
                    </div>          
                    )}
            </div>
        </div>
    );
};

export default DisplayImages;
