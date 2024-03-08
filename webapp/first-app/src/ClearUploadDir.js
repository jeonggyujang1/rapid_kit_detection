import React, { useState } from 'react';
import axios from 'axios';

const ClearUploadDir = () => {
    const [msg, setMsg] = useState(null);

    const handleClear = async () => {
        const response = await axios.get('/clear/');
        setMsg(response.data.message);
	};

    return (
        <div>
          <button onClick={handleClear}>Clear</button>
          <p>{msg}</p>
        </div>
      );
};

export default ClearUploadDir;

