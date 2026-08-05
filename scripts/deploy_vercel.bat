@echo off
echo Deploying Frontend to Vercel Cloud...
cd frontend
npx vercel --prod
pause
