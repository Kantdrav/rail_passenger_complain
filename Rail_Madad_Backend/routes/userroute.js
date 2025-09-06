import express from 'express';


import { register,login,verifyOtp } from '../api/user.js';

const router= express.Router();

router.post("/register",register)
router.post("/otp-veriyfy",verifyOtp)
router.post("/login",login)
export default router;