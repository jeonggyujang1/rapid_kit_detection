import React, { useEffect, useState } from 'react';
import axios from 'axios';
import nullUrl from "./null.webp" 
import Button from '@mui/material/Button';
import Stack from '@mui/material/Stack';

const DisplayImages = () => {
    const [images, setImages] = useState([]);
    const [dir, setDir] = useState('./uploads');
    const [cnts, setCnts] = useState([]);

    useEffect(() => {
        
    });

    const handleCount = async () => {
        const response = await axios.post('/count/',dir);
        if (Array.isArray(response.data)) {
			console.log(Array.isArray(response.data))
            console.log(response.data)
            setCnts(response.data[0])
            setImages(response.data.slice(1));
        } else {
			console.log(Array.isArray(response.data))
			setImages(nullUrl);

        }
    };
    const handleGenerate = async () => {
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
                <Button onClick={handleCount} variant="outlined">
                    Count Kits
                </Button>
                <Button onClick={handleGenerate} variant="outlined">
                    Generate Results
                </Button>
                <Button onClick={changeTargetDir} variant="outlined">
                    Change Target Directory
                </Button>
            </Stack>
            <p>Target Directory : {dir}</p>
            <p># of Kits
                {cnts.map((cnts) => (
                    <li>{cnts}</li>
                ))}
            </p>
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
