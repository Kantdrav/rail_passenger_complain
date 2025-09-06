import cors from 'cors';
import express from 'express';
import mongoose from 'mongoose';
import dotenv from 'dotenv';
import cookieParser from 'cookie-parser';

import userauth from './routes/userroute.js'
import complain from './routes/complaintroute.js'
import testapi from './routes/test.js'
dotenv.config();

const app = express();
const port = process.env.PORT || 3000;

mongoose.set("strictQuery", false);

const connectDB = async () => {
  try {
    await mongoose.connect(process.env.MONGO_URL, { 
      useNewUrlParser: true,
      useUnifiedTopology: true 
    });
    console.log("MongoDB connected successfully");
  } catch (error) {
    console.error("MongoDB connection failed:", error);
  }
};



app.use(cors());
app.use(cookieParser());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.use('/api/railmadad', userauth);
app.use('/api/send/complainto', complain);
app.use('/api/test', testapi);
app.listen(port, () => {
  connectDB();
  console.log(port);
  console.log(`Server is running on port ${port}`);
});

