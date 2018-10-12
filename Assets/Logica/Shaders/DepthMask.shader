Shader "Logica/DepthMask" 
{
	SubShader 
	{
		//Set the mask queue
		Tags { "Queue" = "Geometry-10" }
		
		//Turn off lighting for this shader
		Lighting Off

		ZTest Always
		Zwrite On

		ColorMask 0

		Pass{}
	}
}
