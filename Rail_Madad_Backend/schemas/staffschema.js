import mongoose from "mongoose";

const staffSchema = new mongoose.Schema({
    name: { type: String, required: true },
    mobileNumber: { type: String, required: true },
    department: { type: String, required: true },
    trainNo: { type: String, required: true },
    dutyStart: { type: Date },
    dutyEnd: { type: Date }
});

const Staff = mongoose.models.Staffs || mongoose.model("Staffs", staffSchema);
export default Staff;
